import torch
import torch.nn as nn
import torch.nn.functional as F

class DualAttentionFusion(nn.Module):
    def __init__(self, num_levels=4, in_channels=256):
        super().__init__()
        self.num_levels = num_levels
        self.in_channels = in_channels

        # ----------------------- 通道注意力分支 -----------------------
        self.channel_attn = nn.Sequential(
            nn.Conv2d(in_channels * num_levels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )
        
        # ----------------------- 空间注意力分支 -----------------------
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # ----------------------- 后处理模块 -----------------------
        self.post_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # ----------------------- 权重初始化 -----------------------
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 通道注意力初始化
        nn.init.normal_(self.channel_attn[0].weight, mean=0, std=0.01)
        nn.init.constant_(self.channel_attn[0].bias, 0)
        nn.init.constant_(self.channel_attn[2].weight, 0)
        nn.init.constant_(self.channel_attn[2].bias, 0)
        
        # 空间注意力初始化
        nn.init.normal_(self.spatial_attn[0].weight, mean=0, std=0.01)
        nn.init.constant_(self.spatial_attn[0].bias, 0)

    def forward(self, multi_level_feats):
        """
        Args:
            multi_level_feats: 输入特征张量 
                shape: [num_levels, batch_size, channels, height, width]
                示例: [4, 1, 256, 7, 7]
        Returns:
            fused_feat: 融合后的特征 
                shape: [batch_size, channels, height, width]
                示例: [1, 256, 7, 7]
        """
        # 维度转换 [4,1,256,7,7] => [1,256,4,7,7]
        feats = multi_level_feats.permute(1, 2, 0, 3, 4).contiguous()
        B, C, L, H, W = feats.shape  # B=1, C=256, L=4
        
        # ================= 通道注意力计算 =================
        # 合并层级维度 [1,256,4,7,7] => [1,1024,7,7]
        channel_feats = feats.view(B, C * L, H, W)
        # 全局空间平均池化 [1,1024,1,1]
        gap = channel_feats.mean(dim=(2, 3), keepdim=True)
        # 生成通道权重 [1,256,1,1]
        channel_weights = self.channel_attn(gap)
        # 应用通道注意力 [1,256,4,7,7]
        channel_attned = feats * channel_weights.unsqueeze(2)  # unsqueeze添加层级维度

        # ================= 空间注意力计算 =================
        # 维度转换 [B,C,L,H,W] => [B*L,C,H,W]
        spatial_feats = feats.permute(0, 2, 1, 3, 4).contiguous()  # [1,4,256,7,7]
        spatial_feats = spatial_feats.view(-1, C, H, W)           # [4,256,7,7]
        # 生成空间权重 [4,1,7,7]
        spatial_weights = self.spatial_attn(spatial_feats)
        # 恢复维度 [1,4,1,7,7]
        spatial_weights = spatial_weights.view(B, L, 1, H, W)
        # 维度对齐 [1,1,4,7,7]
        spatial_weights = spatial_weights.permute(0, 2, 1, 3, 4)
        # 应用空间注意力 [1,256,4,7,7]
        spatial_attned = feats * spatial_weights

        # ================= 特征融合 =================
        # 残差连接 [1,256,4,7,7]
        fused = channel_attned + spatial_attned
        # 层级维度求和 [1,256,7,7]
        fused = fused.sum(dim=2)
        # 后处理卷积 [1,256,7,7]
        return self.post_conv(fused)

if __name__ == "__main__":
    # 测试用例
    input_feats = torch.randn(4, 1, 256, 7, 7)  # [层级数, batch, 通道, 高, 宽]
    fusion_module = DualAttentionFusion(num_levels=4)
    output = fusion_module(input_feats)
    print("输入形状:", input_feats.shape)  # torch.Size([4, 1, 256, 7, 7])
    print("输出形状:", output.shape)      # torch.Size([1, 256, 7, 7])