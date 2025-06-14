import math
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
                 activated_experts=2,
                 total_epochs=25):
        super().__init__()
        
        # 基础参数
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.total_epochs = total_epochs
        assert self.head_dim * n_heads == channels, "通道数必须能被头数整除"
        
        # 动态路由参数
        self.moe_experts = moe_experts
        self.activated_experts = activated_experts
        
        # 注册衰减系数缓冲区
        self.register_buffer('alpha', torch.tensor(1.0))
        self.register_buffer('current_epoch', torch.tensor(-1))
        
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

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.route.weight, std=0.02)

    def set_epoch(self, epoch):
        self.current_epoch.fill_(epoch)
        if epoch < 20:
            progress = 0.0  # 前20个epoch进度为0，alpha保持1
        else:
            # 后5个epoch使用非线性加速的进度计算
            adjusted_epoch = epoch - 20
            progress = (adjusted_epoch / 5) ** 3  # 三次方加速
            progress = min(progress, 1.0)  # 限制进度不超过1
        self.alpha.fill_(0.5 * (1 + math.cos(math.pi * progress)))

    def get_rel_pos(self, h=7, w=7):
        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w))
        rel_y = grid_y.reshape(-1, 1) - grid_y.reshape(1, -1) + 6
        rel_x = grid_x.reshape(-1, 1) - grid_x.reshape(1, -1) + 6
        return rel_y.long(), rel_x.long()

    def dynamic_route(self, x):
        if not self.training or self.alpha < 1e-3:
            # 测试模式或完全衰减时返回原始特征
            return x.detach() if self.training else x
            
        b, c, h, w = x.shape
        
        # 带衰减的路由计算
        scores = self.route(x.mean(dim=(2,3))) * self.alpha
        topk_scores, topk_indices = scores.topk(self.activated_experts, dim=1)
        
        softmax_values = F.softmax(topk_scores, dim=1).to(scores.dtype)
        mask = torch.zeros_like(scores).scatter(1, topk_indices, softmax_values)
        
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)
        
        # 记录路由信息
        if self.training:
            self.router_logits = scores.detach()
            self.mask = mask.detach()
            
        return torch.einsum('bm,bmchw->bchw', mask, expert_outs)

    def compute_balance_loss(self, eps=1e-7):
        if not self.training or not hasattr(self, 'router_logits'):
            return torch.tensor(0.0, device=self.alpha.device)
            
        importance = self.mask.sum(dim=0)
        load = (self.router_logits.softmax(dim=1) > 0.1).float().sum(dim=0)
        
        importance = importance + eps
        load = load + eps
        
        balance_loss = self.moe_experts * (importance * load).sum() / (importance.sum() * load.sum())
        return balance_loss * self.alpha  # 衰减平衡损失

    def forward(self, z, x):
        # 带残差的动态路由
        z_base = z.detach() if self.training else z
        z = self.dynamic_route(z) * self.alpha + z_base * (1 - self.alpha)
        
        x_base = x.detach() if self.training else x
        x = self.dynamic_route(x) * self.alpha + x_base * (1 - self.alpha)
        
        # 处理batch维度差异
        if z.size(0) == 1 and x.size(0) > 1:
            z = z.expand(x.size(0), -1, -1, -1)
        
        # 投影到多头空间
        q = self.t_proj(z)
        k, v = self.x_proj(x).chunk(2, dim=1)
        
        # 拆分多头
        q = rearrange(q, 'b (h d) hh ww -> b h (hh ww) d', h=self.n_heads)
        k = rearrange(k, 'b (h d) hh ww -> b h (hh ww) d', h=self.n_heads)
        v = rearrange(v, 'b (h d) hh ww -> b h (hh ww) d', h=self.n_heads)
        
        # 相对位置偏置
        rel_y, rel_x = self.get_rel_pos()
        pos_bias = self.rel_pos_bias[:, rel_y, rel_x]
        
        # 注意力计算
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhqd,bhkd->bhqk', q, k) * scale
        attn += pos_bias.unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        
        # 值聚合
        out = torch.einsum('bhqk,bhkd->bhqd', attn, v)
        out = rearrange(out, 'b h (hh ww) d -> b (h d) hh ww', hh=7, ww=7)
        
        # 最终融合带衰减
        fused = self.fusion(out)
        print("-------alpha---------", self.alpha)
        return fused * self.alpha, 1 - self.alpha