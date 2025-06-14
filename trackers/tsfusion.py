import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops

class DynamicCrossAttention(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        # 定义基础可变形卷积
        self.deform_conv = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=3, 
            padding=1
        )
        # 偏移量预测
        self.offset_conv = nn.Conv2d(
            in_channels*2, 
            2 * 3 * 3,  # 每个位置预测3x3卷积核的(x,y)偏移 
            kernel_size=3, 
            padding=1
        )
        # 权重生成
        self.weight_conv = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, 1),
            nn.Sigmoid()
        )

        self._init_weights()
    def _init_weights(self):
        # 初始化偏移量卷积
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        
        # 初始化权重生成卷积
        nn.init.kaiming_normal_(
            self.weight_conv[0].weight, 
            mode='fan_in',
            nonlinearity='leaky_relu'
        )
        nn.init.constant_(self.weight_conv[0].bias, 0)
        
        # 初始化可变形卷积
        nn.init.kaiming_normal_(
            self.deform_conv.weight, 
            mode='fan_in',
            nonlinearity='leaky_relu'
        )
        nn.init.constant_(self.deform_conv.bias, 0)

    def forward(self, template_feat, search_feat):
        # 特征对齐
        aligned_template = F.interpolate(
            template_feat, 
            size=search_feat.shape[2:], 
            mode='bilinear',
            align_corners=False
        )
        
        # 拼接特征生成参数
        combined = torch.cat([aligned_template, search_feat], dim=1)
        offsets = self.offset_conv(combined)
        weights = self.weight_conv(combined)
        
        # 应用可变形卷积
        fused_feat = ops.deform_conv2d(
            input=search_feat,
            offset=offsets,
            weight=self.deform_conv.weight,
            bias=self.deform_conv.bias,
            mask=weights,
            padding=1,
            stride=1
        )
        return fused_feat

class TrackerWithDynamicInteraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attentions = nn.ModuleList([
            DynamicCrossAttention() for _ in range(4)  # 4个层级
        ])
        
    def forward(self, template_feats, search_feats):
        interacted_feats = []
        for t_feat, s_feat, attn in zip(
            template_feats, search_feats, self.cross_attentions
        ):
            interacted = attn(t_feat, s_feat)
            interacted_feats.append(interacted)
        return interacted_feats