from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Residual-enhanced double convolution block with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + self.shortcut(x))


class FetalHeadUNet(nn.Module):
    """
    U-Net architecture for 3-class Fetal Ultrasound Segmentation
    (0: Background, 1: Pubic Symphysis, 2: Fetal Head).

    Supports:
      - `use_context_stage=False`: Exact v1 architecture for loading preserved
        `FetalHeadUNet-v1.0-seed42_ORIGINAL.pt` weights.
      - `use_context_stage=True`: v2 architecture with a 4th global-context encoder/decoder
        stage and spatial coordinate priors so the receptive field spans the entire
        ultrasound sector.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        base_filters: int = 24,
        use_context_stage: bool = False,
    ) -> None:
        super().__init__()
        self.use_context_stage = use_context_stage
        f1 = base_filters
        f2 = base_filters * 2
        f3 = base_filters * 4
        f4 = base_filters * 8

        eff_in = (in_channels + 2) if use_context_stage else in_channels
        self.enc1 = DoubleConv(eff_in, f1)
        self.enc2 = DoubleConv(f1, f2)
        self.enc3 = DoubleConv(f2, f3)
        self.bottleneck = DoubleConv(f3, f4)

        self.pool = nn.MaxPool2d(2, 2)

        if use_context_stage:
            self.enc4 = DoubleConv(f4, f4)
            self.up4 = nn.ConvTranspose2d(f4, f4, kernel_size=2, stride=2)
            self.dec4 = DoubleConv(f4 + f4, f4)

        self.up3 = nn.ConvTranspose2d(f4, f3, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(f3 + f3, f3)

        self.up2 = nn.ConvTranspose2d(f3, f2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(f2 + f2, f2)

        self.up1 = nn.ConvTranspose2d(f2, f1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(f1 + f1, f1)

        self.seg_head = nn.Conv2d(f1, num_classes, kernel_size=1)

        self.plane_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(f4, f2),
            nn.ReLU(inplace=True),
            nn.Linear(f2, 1),
        )

    def _append_coord_channels(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        device = x.device
        dtype = x.dtype
        yy = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xx = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([x, yy, xx], dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inp = self._append_coord_channels(x) if self.use_context_stage else x
        e1 = self.enc1(inp)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        if self.use_context_stage:
            b_deep = self.enc4(self.pool(b))
            plane_logit = self.plane_head(b_deep).squeeze(-1)
            u4 = self.up4(b_deep)
            if u4.shape[-2:] != b.shape[-2:]:
                u4 = F.interpolate(u4, size=b.shape[-2:], mode="bilinear", align_corners=False)
            b = self.dec4(torch.cat([u4, b], dim=1))
        else:
            plane_logit = self.plane_head(b).squeeze(-1)

        if plane_logit.ndim == 0:
            plane_logit = plane_logit.unsqueeze(0)

        d3 = self.up3(b)
        if d3.shape[-2:] != e3.shape[-2:]:
            d3 = F.interpolate(d3, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        seg_logits = self.seg_head(d1)
        return seg_logits, plane_logit


class StandardPlaneClassifier(nn.Module):
    """
    Dedicated Standard-Plane (SP vs NSP) binary classifier trained and evaluated
    separately from anatomical segmentation so non-standard-plane frames never
    corrupt segmentation encoder BatchNorm statistics or decoder gradients.
    """

    def __init__(self, in_channels: int = 1, base_filters: int = 16) -> None:
        super().__init__()
        f1 = base_filters
        f2 = base_filters * 2
        f3 = base_filters * 4
        f4 = base_filters * 8
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f2, f3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f3),
            nn.ReLU(inplace=True),
            nn.Conv2d(f3, f4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(f4, f2),
            nn.ReLU(inplace=True),
            nn.Linear(f2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(x)).squeeze(-1)
        if logits.ndim == 0:
            logits = logits.unsqueeze(0)
        return logits
