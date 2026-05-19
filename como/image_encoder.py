"""
COMO Image Encoder
==================

Vision backbone (ResNet-50 or Swin Transformer) that encodes molecule
images into spatial feature maps consumed by the Sequence Decoder.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    """
    Image encoder with ResNet-50 or Swin Transformer backbone.
    Outputs spatial feature maps for the Transformer Decoder.
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = False,
        d_model: int = 512
    ):
        super().__init__()
        self.backbone_name = backbone
        self.d_model = d_model
        
        if backbone == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            # Remove avgpool and fc
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            self.feat_dim = 2048

        elif backbone.startswith('swin'):
            if backbone == 'swin_t':
                swin = models.swin_t(weights=models.Swin_T_Weights.DEFAULT if pretrained else None)
                self.feat_dim = 768
            elif backbone == 'swin_s':
                swin = models.swin_s(weights=models.Swin_S_Weights.DEFAULT if pretrained else None)
                self.feat_dim = 768
            elif backbone == 'swin_b':
                swin = models.swin_b(weights=models.Swin_B_Weights.DEFAULT if pretrained else None)
                self.feat_dim = 1024
            elif backbone == 'swin_v2_t':
                swin = models.swin_v2_t(weights=models.Swin_V2_T_Weights.DEFAULT if pretrained else None)
                self.feat_dim = 768
            elif backbone == 'swin_v2_s':
                swin = models.swin_v2_s(weights=models.Swin_V2_S_Weights.DEFAULT if pretrained else None)
                self.feat_dim = 768
            elif backbone == 'swin_v2_b':
                swin = models.swin_v2_b(weights=models.Swin_V2_B_Weights.DEFAULT if pretrained else None)
                self.feat_dim = 1024
            else:
                raise ValueError(f"Unknown Swin variant: {backbone}")
            
            self.backbone = swin.features
            self.norm = swin.norm
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Project to d_model
        self.proj = nn.Conv2d(self.feat_dim, d_model, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] RGB images
        Returns:
            [B, d_model, H', W'] feature maps
        """
        if self.backbone_name == 'resnet50':
            x = self.backbone(x)
        else:
            x = self.backbone(x)
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2).contiguous()
        
        x = self.proj(x)
        return x


# ======================== Positional Encoding ========================

import numpy as np


class PositionalEncoding2D(nn.Module):
    """2D sinusoidal positional encoding for image features."""
    
    def __init__(self, d_model: int, max_h: int = 100, max_w: int = 100):
        super().__init__()
        self.d_model = d_model
        
        # d_model must be divisible by 4 (split among y_sin, y_cos, x_sin, x_cos)
        assert d_model % 4 == 0, f"d_model ({d_model}) must be divisible by 4"
        
        pe = torch.zeros(max_h, max_w, d_model)
        
        y_pos = torch.arange(0, max_h).unsqueeze(1).float()  # [max_h, 1]
        x_pos = torch.arange(0, max_w).unsqueeze(1).float()  # [max_w, 1]
        
        # Each axis gets d_model/4 frequency components
        dim_per_axis = d_model // 4
        div_term = torch.exp(torch.arange(0, dim_per_axis) * -(np.log(10000.0) / dim_per_axis))
        
        # Compute sinusoidal encodings per axis
        y_sin = torch.sin(y_pos * div_term)  # [max_h, dim_per_axis]
        y_cos = torch.cos(y_pos * div_term)  # [max_h, dim_per_axis]
        x_sin = torch.sin(x_pos * div_term)  # [max_w, dim_per_axis]
        x_cos = torch.cos(x_pos * div_term)  # [max_w, dim_per_axis]
        
        # Assemble [max_h, max_w, d_model] encoding table
        for i in range(max_h):
            for j in range(max_w):
                pe[i, j, 0*dim_per_axis:1*dim_per_axis] = y_sin[i]
                pe[i, j, 1*dim_per_axis:2*dim_per_axis] = y_cos[i]
                pe[i, j, 2*dim_per_axis:3*dim_per_axis] = x_sin[j]
                pe[i, j, 3*dim_per_axis:4*dim_per_axis] = x_cos[j]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, d_model, H, W]
        Returns:
            x + positional encoding: [B, d_model, H, W]
        """
        B, C, H, W = x.shape
        pe_tensor: torch.Tensor = self.pe
        pe = pe_tensor[:H, :W, :].permute(2, 0, 1).unsqueeze(0)  # [1, d_model, H, W]
        return x + pe
