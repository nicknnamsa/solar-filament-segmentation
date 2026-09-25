"""EdgeAttNet: a U-Net with edge-guided self-attention at the bottleneck.

Reimplemented from the architecture description in "EdgeAttNet: Towards Barb-Aware Filament Segmentation"
(arXiv:2509.02964), NOT from the authors' released weights (github.com/dasjar/EdgeAttNet) -- those were trained
on a split of MAGFiLO that almost certainly overlaps with our competition's held-out test set (their reported
1,295/45/99 split of 1,439 filtered images vs. our 601/106 of 707, both drawn from the same MAGFiLO pool). We
reimplement the architecture only and train it fresh on our own permitted training split.

Unlike everything else in this project (Mask R-CNN, instance segmentation), this is semantic segmentation: the
model outputs one per-pixel filament-probability map per image, with no notion of separate instances. See
edgeattnet_predict.py for how that gets turned into individual filament instances with scores, so it can plug
into the same evaluate.py / predict.py / submission pipeline as everything else.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(3x3 conv -> BN -> ReLU) x2, the standard U-Net building block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EdgeGuidedAttention(nn.Module):
    """Multi-head self-attention at the bottleneck, with a learnable edge prior added into Q and K (not V).

    E = sigmoid(3x3 conv(raw image)), resized to the bottleneck's spatial size and projected to its channel
    count. Q = K = flattened_features + edge_prior, V = flattened_features. Paper uses 4 heads x 128 dim (512
    total, matching a 512-channel bottleneck).
    """

    def __init__(self, channels, num_heads=4, head_dim=128):
        super().__init__()
        assert num_heads * head_dim == channels, "num_heads * head_dim must equal the bottleneck channel count"
        self.num_heads, self.head_dim = num_heads, head_dim
        self.edge_conv = nn.Conv2d(1, 1, 3, padding=1)          # E = sigmoid(conv(image)); done once per image
        self.edge_proj = nn.Conv2d(1, channels, 1)              # project the (resized) edge map to `channels`
        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def edge_prior(self, image):
        return torch.sigmoid(self.edge_conv(image))

    def forward(self, x, edge_map):
        b, c, h, w = x.shape
        e = F.interpolate(edge_map, size=(h, w), mode="bilinear", align_corners=False)
        e = self.edge_proj(e)
        x_flat = x.flatten(2).transpose(1, 2)          # (B, H*W, C)
        e_flat = e.flatten(2).transpose(1, 2)
        q = k = x_flat + e_flat
        v = x_flat
        out, _ = self.mha(q, k, v)
        return out.transpose(1, 2).reshape(b, c, h, w)


class EdgeAttNet(nn.Module):
    """4-stage U-Net (channels 64/128/256/512), EACH stage followed by its own 2x2 maxpool -- 16x spatial
    reduction by the time the bottleneck is reached, matching the paper. Edge-guided attention is applied
    directly on that 16x-downsampled, 512-channel feature map.

    Input: (B, 1, H, W) grayscale, values in [0, 1]. Output: (B, 1, H, W) logits (apply sigmoid for probabilities).
    H and W must be divisible by 16 (four 2x downsamples).
    """

    CHANNELS = [64, 128, 256, 512]

    def __init__(self, in_channels=1, num_heads=4, head_dim=128):
        super().__init__()
        c1, c2, c3, c4 = self.CHANNELS
        self.enc1 = DoubleConv(in_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)                  # 4th stage -> 16x reduction after its pool
        self.pool = nn.MaxPool2d(2)
        self.attn = EdgeGuidedAttention(c4, num_heads, head_dim)

        self.up4 = nn.ConvTranspose2d(c4, c4, 2, stride=2)
        self.dec4 = DoubleConv(c4 + c4, c3)             # skip from enc4 (post-attention path) + upsampled
        self.up3 = nn.ConvTranspose2d(c3, c3, 2, stride=2)
        self.dec3 = DoubleConv(c3 + c3, c2)
        self.up2 = nn.ConvTranspose2d(c2, c2, 2, stride=2)
        self.dec2 = DoubleConv(c2 + c2, c1)
        self.up1 = nn.ConvTranspose2d(c1, c1, 2, stride=2)
        self.dec1 = DoubleConv(c1 + c1, c1)
        self.out_conv = nn.Conv2d(c1, 1, 1)

    def forward(self, x):
        assert x.shape[-1] % 16 == 0 and x.shape[-2] % 16 == 0, "H and W must be divisible by 16"
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))                   # 16x down from here on

        edge_map = self.attn.edge_prior(x)               # computed once from the raw image, at input resolution
        b = self.attn(self.pool(e4), edge_map)           # pool e4 too -- this is the actual 16x-downsampled bottleneck

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


def bce_dice_loss(logits, target, smooth=1.0):
    """L = L_BCE + L_Dice, as in the paper."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1 - ((2 * intersection + smooth) / (union + smooth)).mean()
    return bce + dice
