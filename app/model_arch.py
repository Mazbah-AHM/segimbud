import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class FastKANLayer(nn.Module):
    def __init__(self, channels, grid_size=5, spline_orders=None):
        super().__init__()
        spline_orders = spline_orders or [1, 2, 3]
        self.grid = nn.Parameter(torch.linspace(-1, 1, grid_size), requires_grad=False)
        self.spline_orders = spline_orders
        self.spline_weights = nn.ParameterList(
            [nn.Parameter(torch.randn(channels, grid_size) * 0.01) for _ in spline_orders]
        )
        self.channel_scales = nn.Parameter(torch.ones(channels))
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x)
        x_flat = torch.tanh(x.view(b, c, -1))
        out = torch.zeros(b, c, h * w, device=x.device)

        for order, weights in zip(self.spline_orders, self.spline_weights):
            diff = x_flat.unsqueeze(-1) - self.grid.unsqueeze(0).unsqueeze(0)
            basis = F.relu(1 - diff.abs()) ** order
            out += torch.einsum("bchw,cg->bch", basis, weights)

        out = out * self.channel_scales.view(1, c, 1)
        return residual + out.view(b, c, h, w) * 0.1


class MultiScaleSelfAttention(nn.Module):
    def __init__(self, channels, scales=None):
        super().__init__()
        self.attention_channels = max(channels // 8, 1)
        self.query_conv = nn.Conv2d(channels, self.attention_channels, 1)
        self.key_conv = nn.Conv2d(channels, self.attention_channels, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(channels, channels, 1)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        b, c, h, w = x.size()
        q = self.query_conv(x).view(b, -1, h * w).transpose(1, 2)
        k = self.key_conv(x).view(b, -1, h * w).transpose(1, 2)
        v = self.value_conv(x).view(b, -1, h * w)
        attn = self.softmax(q @ k.transpose(-1, -2) / self.attention_channels ** 0.5)
        out = attn @ v.transpose(-1, -2)
        out = out.transpose(-1, -2).view(b, c, h, w)
        out = self.out_conv(out)
        out = self.norm(out)
        return x + out * 0.1


class EffKANSeg(nn.Module):
    def __init__(self, num_classes=7, encoder_name="efficientnet-b3"):
        super().__init__()
        self.base = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
            activation=None,
        )
        self.encoder = self.base.encoder
        self.decoder = self.base.decoder
        self.segmentation_head = self.base.segmentation_head
        self.bottleneck = nn.Sequential(
            FastKANLayer(384, grid_size=5, spline_orders=[1, 2, 3]),
            MultiScaleSelfAttention(384, scales=[1, 2, 4]),
            nn.Conv2d(384, 384, 3, padding=1, groups=384),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc = self.encoder(x)
        enc[-1] = self.bottleneck(enc[-1])
        decoded = self.decoder(*enc)
        return self.segmentation_head(decoded)
