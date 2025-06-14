import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MLASimilarity(nn.Module):
    def __init__(self, 
                 channels=256,
                 n_heads=8,
                 qk_rank=64,
                 kv_rank=128,
                 moe_experts=4,
                 activated_experts=2):
        super().__init__()
        
        # 基础参数
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        assert self.head_dim * n_heads == channels, "通道数必须能被头数整除"
        
        # 动态路由参数
        self.moe_experts = moe_experts
        self.activated_experts = activated_experts
        
        # 相对位置编码表
        self.rel_pos_bias = nn.Parameter(torch.zeros(n_heads, 13, 13))
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)
        
        # 投影层
        self.t_proj = nn.Sequential(
            nn.Conv2d(channels, qk_rank, 3, padding=1),
            nn.GroupNorm(4, qk_rank),
            nn.Conv2d(qk_rank, n_heads*self.head_dim, 1)
        )
        
        self.x_proj = nn.Sequential(
            nn.Conv2d(channels, kv_rank, 3, padding=1),
            nn.GroupNorm(4, kv_rank),
            nn.Conv2d(kv_rank, 2*n_heads*self.head_dim, 1)
        )
        
        # 动态路由层
        self.route = nn.Linear(channels, moe_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels//2, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(channels//2, channels, 1)
            ) for _ in range(moe_experts)]
        )
        
        # 输出融合
        self.fusion = nn.Sequential(
            nn.Conv2d(n_heads*self.head_dim, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.route.weight, std=0.02)  # 路由层初始化

    def get_rel_pos(self, h=7, w=7):
        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w))
        rel_y = grid_y.reshape(-1, 1) - grid_y.reshape(1, -1) + 6  # [49,49]
        rel_x = grid_x.reshape(-1, 1) - grid_x.reshape(1, -1) + 6
        return rel_y.long(), rel_x.long()

    def dynamic_route(self, x):
        b, c, h, w = x.shape
        # 路由分数计算
        scores = self.route(x.mean(dim=(2,3)))  # [b, moe]
        topk_scores, topk_indices = scores.topk(self.activated_experts, dim=1)  # [b, k]
        
        # 类型对齐修正
        softmax_values = F.softmax(topk_scores, dim=1).to(scores.dtype)  # 确保类型一致
        mask = torch.zeros_like(scores).scatter(1, topk_indices, softmax_values)
        
        # 并行计算专家输出
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)  # [b, moe, c, h, w]
        
        # 收集路由统计信息
        self.router_logits = scores  # 用于计算负载均衡损失
        self.mask = mask  # 用于计算负载均衡损失
        
        return torch.einsum('bm,bmchw->bchw', mask, expert_outs)

    def compute_balance_loss(self, eps=1e-7):
        if not hasattr(self, 'router_logits') or not hasattr(self, 'mask'):
            return torch.tensor(0.0, device=self.rel_pos_bias.device)  # 确保返回tensor
        
        importance = self.mask.sum(dim=0)
        load = (self.router_logits.softmax(dim=1) > 0.1).float().sum(dim=0)
        
        # 添加维度检查
        if importance.dim() == 0:
            importance = importance.unsqueeze(0)
        if load.dim() == 0:
            load = load.unsqueeze(0)
        
        balance_loss = self.moe_experts * (importance * load).sum() / (importance.sum() * load.sum() + eps)
        return balance_loss

    def forward(self, z, x):
        # 动态路由增强
        z = self.dynamic_route(z)
        x = self.dynamic_route(x)
        
        # 处理batch维度差异
        if z.size(0) == 1 and x.size(0) > 1:
            z = z.expand(x.size(0), -1, -1, -1)  # 扩展模板到匹配batch size
        
        # 投影到多头空间
        q = self.t_proj(z)
        k, v = self.x_proj(x).chunk(2, dim=1)
        
        # 拆分多头
        q = rearrange(q, 'b (h d) hh ww -> b h (hh ww) d', h=self.n_heads)
        k = rearrange(k, 'b (h d) hh ww -> b h (hh ww) d', h=self.n_heads)
        v = rearrange(v, 'b (h d) hh ww -> b h (hh ww) d', h=self.n_heads)
        
        # 相对位置偏置
        rel_y, rel_x = self.get_rel_pos()
        pos_bias = self.rel_pos_bias[:, rel_y, rel_x]  # [h,49,49]
        
        # 注意力计算
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhqd,bhkd->bhqk', q, k) * scale
        attn += pos_bias.unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        
        # 值聚合
        out = torch.einsum('bhqk,bhkd->bhqd', attn, v)
        out = rearrange(out, 'b h (hh ww) d -> b (h d) hh ww', hh=7, ww=7)
        out = self.fusion(out)
        
        return out