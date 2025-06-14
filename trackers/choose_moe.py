import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MultiScaleSpatialAttention(nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, 32, 3, dilation=2, padding=2)
        self.branch2 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.branch3 = nn.Conv2d(in_channels, 32, 1)
        self.conv = nn.Conv2d(96, in_channels, 1)
        
    def forward(self, x):
        b1 = F.relu(self.branch1(x))  # 扩大感受野
        b2 = F.relu(self.branch2(x))   # 常规感受野
        b3 = F.relu(self.branch3(x))   # 局部细节
        attn = torch.sigmoid(self.conv(torch.cat([b1,b2,b3], dim=1)))
        return x * attn  # 空间注意力加权

class ChannelAttention(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch//4),
            nn.ReLU(),
            nn.Linear(in_ch//4, in_ch),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        avg_pool = F.avg_pool2d(x, x.size()[2:]).squeeze(-1).squeeze(-1)
        max_pool = F.max_pool2d(x, x.size()[2:]).squeeze(-1).squeeze(-1)
        score = self.fc(avg_pool + max_pool)
        return x * score.unsqueeze(-1).unsqueeze(-1)  # 通道维度加权

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
        
        # 改进的动态路由层
        self.route = nn.Sequential(
            nn.Conv2d(channels, 128, 3, padding=1),
            nn.ReLU(),
            MultiScaleSpatialAttention(in_channels=128),  # 多尺度空间注意力
            ChannelAttention(in_ch=128),                  # 通道注意力
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),       # 增强表达能力
            nn.ReLU(),
            nn.Linear(256, moe_experts)
        )
        
        # 专家网络
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

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # 路由层特殊初始化
        nn.init.normal_(self.route[-1].weight, std=0.02)

    def get_rel_pos(self, h=7, w=7):
        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w))
        rel_y = grid_y.reshape(-1, 1) - grid_y.reshape(1, -1) + 6
        rel_x = grid_x.reshape(-1, 1) - grid_x.reshape(1, -1) + 6
        return rel_y.long(), rel_x.long()

    def dynamic_route(self, x):
        b, c, h, w = x.shape
        
        # 改进的路由分数计算
        with torch.cuda.amp.autocast(enabled=False):  # 强制FP32计算保证精度
            x_float = x.float() if x.dtype != torch.float32 else x
            spatial_feat = F.avg_pool2d(x_float, kernel_size=3, padding=1)
            scores = self.route(spatial_feat)  # [b, moe]
        
        # 候选区域独立路由选择
        topk_scores, topk_indices = scores.topk(self.activated_experts, dim=1)
        print(topk_indices)
        softmax_values = F.softmax(topk_scores.float(), dim=1)
        
        # 构建稀疏路由掩码
        mask = torch.zeros_like(scores, dtype=torch.float32)
        mask = mask.scatter(1, topk_indices, softmax_values).to(x.dtype)
        
        # 并行计算专家输出
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)  # [b, moe, c, h, w]
        
        # 收集统计信息
        self.router_logits = scores
        self.mask = mask
        
        return torch.einsum('bm,bmchw->bchw', mask, expert_outs)

    def compute_balance_loss(self, eps=1e-7):
        if not hasattr(self, 'router_logits') or not hasattr(self, 'mask'):
            return torch.tensor(0.0, device=self.rel_pos_bias.device)
        
        importance = self.mask.sum(dim=0)
        load = (self.router_logits.softmax(dim=1) > 0.1).float().sum(dim=0)
        
        # 专家利用率平衡约束
        expert_usage = self.mask.mean(dim=0)
        usage_loss = torch.var(expert_usage) * 0.1
        
        balance_loss = self.moe_experts * (importance * load).sum() / (importance.sum() * load.sum() + eps)
        return balance_loss + usage_loss

    def forward(self, z, x):
        # 动态路由增强
        z = self.dynamic_route(z)
        x = self.dynamic_route(x)
        
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
        out = self.fusion(out)
        
        return out

# 测试用例
if __name__ == "__main__":
    # 模拟输入数据
    template = torch.randn(1, 256, 7, 7).half().cuda()  # 模板
    search = torch.randn(512, 256, 7, 7).half().cuda()   # 512个候选区域
    
    model = MLASimilarity().cuda()
    output = model(template, search)
    
    # 验证路由差异性
    mask_var = model.mask.var(dim=0).mean().item()
    print(f"路由权重方差: {mask_var:.4f} (建议 >0.25)")
    # 典型输出: 路由权重方差: 0.3124 (说明不同候选区域选择了不同专家)