import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.smt import smt_t


# ============================================================
# Basic Blocks
# ============================================================

class ConvBNReLU(nn.Module):
    """Convolution + BatchNorm + ReLU."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualConvBlock(nn.Module):
    """Two-layer residual convolution block."""

    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.block(x))


# ============================================================
# Haar Wavelet Transform
# ============================================================

class HaarDWT(nn.Module):
    """Haar discrete wavelet transform."""

    def __init__(self):
        super().__init__()

        ll = torch.tensor(
            [[0.5, 0.5],
             [0.5, 0.5]],
            dtype=torch.float32,
        )

        lh = torch.tensor(
            [[-0.5, 0.5],
             [-0.5, 0.5]],
            dtype=torch.float32,
        )

        hl = torch.tensor(
            [[-0.5, -0.5],
             [0.5, 0.5]],
            dtype=torch.float32,
        )

        hh = torch.tensor(
            [[0.5, -0.5],
             [-0.5, 0.5]],
            dtype=torch.float32,
        )

        filters = torch.stack(
            [ll, lh, hl, hh],
            dim=0,
        ).unsqueeze(1)

        self.register_buffer("filters", filters)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        pad_h = height % 2
        pad_w = width % 2

        if pad_h != 0 or pad_w != 0:
            x = F.pad(
                x,
                (0, pad_w, 0, pad_h),
                mode="replicate",
            )

        filters = self.filters.to(
            device=x.device,
            dtype=x.dtype,
        ).repeat(channels, 1, 1, 1)

        output = F.conv2d(
            x,
            filters,
            stride=2,
            padding=0,
            groups=channels,
        )

        output = output.reshape(
            batch_size,
            channels,
            4,
            output.shape[-2],
            output.shape[-1],
        )

        ll = output[:, :, 0]
        lh = output[:, :, 1]
        hl = output[:, :, 2]
        hh = output[:, :, 3]

        return ll, (lh, hl, hh)


# ============================================================
# Progressive Wavelet Boundary Model
# ============================================================

class WaveletFusionBlock(nn.Module):
    """Fuse high-frequency wavelet responses with encoder features."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, residual):
        x = self.fusion(x)
        x = x + residual
        return self.activation(x)


class ProgressiveWaveletBoundaryModel(nn.Module):
    """
    Progressive Wavelet Boundary Model (PWBM).

    x1 -> wf1 -> wf2 -> wf3
    """

    def __init__(self, channels=48):
        super().__init__()

        self.dwt = HaarDWT()

        self.wavelet_fuse1 = WaveletFusionBlock(
            in_channels=channels * 4,
            out_channels=channels,
        )

        self.wavelet_fuse2 = WaveletFusionBlock(
            in_channels=channels * 4,
            out_channels=channels,
        )

        self.wavelet_fuse3 = WaveletFusionBlock(
            in_channels=channels * 4,
            out_channels=channels,
        )

    @staticmethod
    def resize_like(x, reference):
        if x.shape[-2:] != reference.shape[-2:]:
            x = F.interpolate(
                x,
                size=reference.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return x

    def extract_high_frequency(self, x, reference):
        _, (lh, hl, hh) = self.dwt(x)

        high_frequency = torch.cat(
            [lh, hl, hh],
            dim=1,
        )

        high_frequency = self.resize_like(
            high_frequency,
            reference,
        )

        return high_frequency

    def forward(self, x1, x2, x3, x4):

        # Stage 1
        high_freq1 = self.extract_high_frequency(
            x1,
            x2,
        )

        wavelet_feat1 = torch.cat(
            [x2, high_freq1],
            dim=1,
        )

        wavelet_feat1 = self.wavelet_fuse1(
            wavelet_feat1,
            residual=x2,
        )

        # Stage 2
        high_freq2 = self.extract_high_frequency(
            wavelet_feat1,
            x3,
        )

        wavelet_feat2 = torch.cat(
            [x3, high_freq2],
            dim=1,
        )

        wavelet_feat2 = self.wavelet_fuse2(
            wavelet_feat2,
            residual=x3,
        )

        # Stage 3
        high_freq3 = self.extract_high_frequency(
            wavelet_feat2,
            x4,
        )

        wavelet_feat3 = torch.cat(
            [x4, high_freq3],
            dim=1,
        )

        wavelet_feat3 = self.wavelet_fuse3(
            wavelet_feat3,
            residual=x4,
        )

        return (
            wavelet_feat1,
            wavelet_feat2,
            wavelet_feat3,
        )


# ============================================================
# Wavelet-guided Criss-Cross Attention
# ============================================================

class WaveletGuidedCrissCrossAttention(nn.Module):
    """Wavelet-guided Criss-Cross Attention (WCCA)."""

    def __init__(
        self,
        channels=48,
        reduction=8,
    ):
        super().__init__()

        reduced_channels = max(
            channels // reduction,
            1,
        )

        self.query_conv = nn.Conv2d(
            channels,
            reduced_channels,
            kernel_size=1,
        )

        self.key_conv = nn.Conv2d(
            channels,
            reduced_channels,
            kernel_size=1,
        )

        self.value_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
        )

        self.edge_gate = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                1,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

        self.output_projection = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )

        self.gamma = nn.Parameter(
            torch.tensor(0.1)
        )

        self.softmax = nn.Softmax(dim=3)
        self.scale = reduced_channels ** -0.5

    @staticmethod
    def build_diagonal_mask(
        batch_size,
        height,
        width,
        device,
        dtype,
    ):
        mask = torch.full(
            (height,),
            float("inf"),
            device=device,
            dtype=dtype,
        )

        mask = -torch.diag(mask)

        return mask.unsqueeze(0).repeat(
            batch_size * width,
            1,
            1,
        )

    def forward(self, wavelet_feat, semantic_feat):

        if wavelet_feat.shape[-2:] != semantic_feat.shape[-2:]:
            wavelet_feat = F.interpolate(
                wavelet_feat,
                size=semantic_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        batch_size, channels, height, width = semantic_feat.shape

        query = self.query_conv(wavelet_feat)
        key = self.key_conv(semantic_feat)
        value = self.value_conv(semantic_feat)

        # ----------------------------------------------------
        # Vertical attention
        # ----------------------------------------------------

        query_h = query.permute(
            0, 3, 1, 2
        ).contiguous()

        query_h = query_h.view(
            batch_size * width,
            -1,
            height,
        ).permute(0, 2, 1)

        key_h = key.permute(
            0, 3, 1, 2
        ).contiguous()

        key_h = key_h.view(
            batch_size * width,
            -1,
            height,
        )

        value_h = value.permute(
            0, 3, 1, 2
        ).contiguous()

        value_h = value_h.view(
            batch_size * width,
            -1,
            height,
        )

        energy_h = torch.bmm(
            query_h,
            key_h,
        ) * self.scale

        energy_h = energy_h + self.build_diagonal_mask(
            batch_size,
            height,
            width,
            semantic_feat.device,
            energy_h.dtype,
        )

        energy_h = energy_h.view(
            batch_size,
            width,
            height,
            height,
        ).permute(0, 2, 1, 3)

        # ----------------------------------------------------
        # Horizontal attention
        # ----------------------------------------------------

        query_w = query.permute(
            0, 2, 1, 3
        ).contiguous()

        query_w = query_w.view(
            batch_size * height,
            -1,
            width,
        ).permute(0, 2, 1)

        key_w = key.permute(
            0, 2, 1, 3
        ).contiguous()

        key_w = key_w.view(
            batch_size * height,
            -1,
            width,
        )

        value_w = value.permute(
            0, 2, 1, 3
        ).contiguous()

        value_w = value_w.view(
            batch_size * height,
            -1,
            width,
        )

        energy_w = torch.bmm(
            query_w,
            key_w,
        ) * self.scale

        energy_w = energy_w.view(
            batch_size,
            height,
            width,
            width,
        )

        # ----------------------------------------------------
        # Joint normalization
        # ----------------------------------------------------

        attention = self.softmax(
            torch.cat(
                [energy_h, energy_w],
                dim=3,
            )
        )

        attention_h = attention[:, :, :, :height]

        attention_h = attention_h.permute(
            0, 2, 1, 3
        ).contiguous()

        attention_h = attention_h.view(
            batch_size * width,
            height,
            height,
        )

        attention_w = attention[
            :, :, :, height:height + width
        ]

        attention_w = attention_w.contiguous().view(
            batch_size * height,
            width,
            width,
        )

        # ----------------------------------------------------
        # Context aggregation
        # ----------------------------------------------------

        context_h = torch.bmm(
            value_h,
            attention_h.permute(0, 2, 1),
        )

        context_h = context_h.view(
            batch_size,
            width,
            channels,
            height,
        ).permute(0, 2, 3, 1)

        context_w = torch.bmm(
            value_w,
            attention_w.permute(0, 2, 1),
        )

        context_w = context_w.view(
            batch_size,
            height,
            channels,
            width,
        ).permute(0, 2, 1, 3)

        context = context_h + context_w
        context = self.output_projection(context)

        # ----------------------------------------------------
        # Wavelet edge gate
        # ----------------------------------------------------

        gate = self.edge_gate(wavelet_feat)

        guided_feature = (
            semantic_feat
            + self.gamma * gate * context
        )

        return guided_feature


# ============================================================
# Wavelet Boundary-guided Difference Fusion
# ============================================================

class WaveletBoundaryGuidedDifferenceFusion(nn.Module):
    """Wavelet Boundary-guided Difference Fusion (WBDF)."""

    def __init__(self, channels=48):
        super().__init__()

        self.wcca = WaveletGuidedCrissCrossAttention(
            channels=channels,
            reduction=8,
        )

        self.fusion = nn.Sequential(
            ConvBNReLU(
                channels * 3,
                channels,
                kernel_size=3,
                padding=1,
            ),
            ResidualConvBlock(channels),
        )

        self.gamma = nn.Parameter(
            torch.tensor(0.1)
        )

    def forward(self, semantic_feat, wavelet_feat):

        guided_feat = self.wcca(
            wavelet_feat,
            semantic_feat,
        )

        difference_feat = torch.abs(
            guided_feat - semantic_feat
        )

        fusion_feat = torch.cat(
            [
                semantic_feat,
                guided_feat,
                difference_feat,
            ],
            dim=1,
        )

        fusion_feat = self.fusion(
            fusion_feat
        )

        output = (
            semantic_feat
            + self.gamma * fusion_feat
        )

        return output


# ============================================================
# Decoder
# ============================================================

class MultiScaleContextBlock(nn.Module):
    """Multi-scale context aggregation."""

    def __init__(self, channels=48):
        super().__init__()

        self.branch1 = ConvBNReLU(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            dilation=1,
        )

        self.branch3 = ConvBNReLU(
            channels,
            channels,
            kernel_size=3,
            padding=3,
            dilation=3,
        )

        self.branch5 = ConvBNReLU(
            channels,
            channels,
            kernel_size=3,
            padding=5,
            dilation=5,
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                channels * 3,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            ResidualConvBlock(channels),
        )

    def forward(self, x):

        feat1 = self.branch1(x)
        feat3 = self.branch3(x)
        feat5 = self.branch5(x)

        output = torch.cat(
            [feat1, feat3, feat5],
            dim=1,
        )

        return self.fusion(output)


class UpFusionBlock(nn.Module):
    """Top-down feature fusion block."""

    def __init__(self, channels=48):
        super().__init__()

        self.fusion = nn.Sequential(
            ConvBNReLU(
                channels * 2,
                channels,
                kernel_size=3,
                padding=1,
            ),
            ResidualConvBlock(channels),
        )

    def forward(self, high_level, skip):

        high_level = F.interpolate(
            high_level,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        output = torch.cat(
            [high_level, skip],
            dim=1,
        )

        return self.fusion(output)


class PredictionHead(nn.Module):
    """Object prediction head."""

    def __init__(
        self,
        channels=48,
        out_channels=1,
    ):
        super().__init__()

        self.head = nn.Sequential(
            ConvBNReLU(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Conv2d(
                channels,
                out_channels,
                kernel_size=1,
            ),
        )

    def forward(self, x, output_size=None):

        prediction = self.head(x)

        if output_size is not None:
            prediction = F.interpolate(
                prediction,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

        return prediction


class WaveletCODDecoder(nn.Module):

    def __init__(
        self,
        channels=48,
        out_channels=1,
    ):
        super().__init__()

        self.context = MultiScaleContextBlock(
            channels
        )

        self.up4_to_3 = UpFusionBlock(channels)
        self.up3_to_2 = UpFusionBlock(channels)
        self.up2_to_1 = UpFusionBlock(channels)

        self.pred4 = PredictionHead(
            channels,
            out_channels,
        )

        self.pred3 = PredictionHead(
            channels,
            out_channels,
        )

        self.pred2 = PredictionHead(
            channels,
            out_channels,
        )

        self.pred1 = PredictionHead(
            channels,
            out_channels,
        )

    def forward(
        self,
        x1,
        x2,
        x3,
        x4,
        output_size=None,
    ):

        d4 = self.context(x4)
        d3 = self.up4_to_3(d4, x3)
        d2 = self.up3_to_2(d3, x2)
        d1 = self.up2_to_1(d2, x1)

        pred4 = self.pred4(
            d4,
            output_size,
        )

        pred3 = self.pred3(
            d3,
            output_size,
        )

        pred2 = self.pred2(
            d2,
            output_size,
        )

        pred1 = self.pred1(
            d1,
            output_size,
        )

        return pred1, pred2, pred3, pred4


# ============================================================
# Auxiliary Edge Prediction
# ============================================================

class EdgeRefinementHead(nn.Module):

    def __init__(self, channels=48):
        super().__init__()

        self.head = nn.Sequential(
            ConvBNReLU(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),

            ResidualConvBlock(channels),

            ConvBNReLU(
                channels,
                channels // 2,
                kernel_size=3,
                padding=1,
            ),

            nn.Conv2d(
                channels // 2,
                1,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        return self.head(x)


# ============================================================
# Backbone Loading
# ============================================================

def load_pretrained_backbone(
    backbone,
    checkpoint_path,
):
    """Load SMT pretrained weights."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if isinstance(checkpoint, dict):

        if "model" in checkpoint:
            checkpoint = checkpoint["model"]

        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        elif "net" in checkpoint:
            checkpoint = checkpoint["net"]

    model_state = backbone.state_dict()
    pretrained_state = {}

    for key, value in checkpoint.items():

        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        if new_key.startswith("backbone."):
            new_key = new_key[len("backbone."):]

        if (
            new_key in model_state
            and model_state[new_key].shape == value.shape
        ):
            pretrained_state[new_key] = value

    model_state.update(pretrained_state)
    backbone.load_state_dict(model_state)

    return backbone


# ============================================================
# Full Network
# ============================================================

class WaveletBoundaryCODNet(nn.Module):

    def __init__(
        self,
        channels=48,
        backbone_checkpoint=None,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.backbone = smt_t()

        if backbone_checkpoint is not None:
            self.backbone = load_pretrained_backbone(
                self.backbone,
                backbone_checkpoint,
            )

        self.proj1 = ConvBNReLU(
            64,
            channels,
            kernel_size=1,
            padding=0,
        )

        self.proj2 = ConvBNReLU(
            128,
            channels,
            kernel_size=1,
            padding=0,
        )

        self.proj3 = ConvBNReLU(
            256,
            channels,
            kernel_size=1,
            padding=0,
        )

        self.proj4 = ConvBNReLU(
            512,
            channels,
            kernel_size=1,
            padding=0,
        )

        # ----------------------------------------------------
        # PWBM
        # ----------------------------------------------------

        self.pwbm = ProgressiveWaveletBoundaryModel(
            channels=channels
        )

        # ----------------------------------------------------
        # WBDF
        # ----------------------------------------------------

        self.wbdf2 = WaveletBoundaryGuidedDifferenceFusion(
            channels
        )

        self.wbdf3 = WaveletBoundaryGuidedDifferenceFusion(
            channels
        )

        self.wbdf4 = WaveletBoundaryGuidedDifferenceFusion(
            channels
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.decoder = WaveletCODDecoder(
            channels=channels,
            out_channels=1,
        )

        # ----------------------------------------------------
        # Edge branch
        # ----------------------------------------------------

        self.edge_head = EdgeRefinementHead(
            channels
        )

    def extract_features(self, x):

        features = self.backbone(x)

        x1 = self.proj1(features[0])
        x2 = self.proj2(features[1])
        x3 = self.proj3(features[2])
        x4 = self.proj4(features[3])

        return x1, x2, x3, x4

    def forward(self, x):

        input_size = x.shape[-2:]

        # Encoder
        x1, x2, x3, x4 = self.extract_features(x)

        # PWBM
        wf1, wf2, wf3 = self.pwbm(
            x1,
            x2,
            x3,
            x4,
        )

        # WBDF
        x2 = self.wbdf2(
            x2,
            wf1,
        )

        x3 = self.wbdf3(
            x3,
            wf2,
        )

        x4 = self.wbdf4(
            x4,
            wf3,
        )

        # Decoder
        pred1, pred2, pred3, pred4 = self.decoder(
            x1,
            x2,
            x3,
            x4,
            output_size=input_size,
        )

        # Auxiliary edge prediction
        edge_pred = self.edge_head(wf1)

        edge_pred = F.interpolate(
            edge_pred,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        # Keep the original output order for compatibility:
        # pred3, pred2, pred1, edge_pred, pred4
        return (
            pred3,
            pred2,
            pred1,
            edge_pred,
            pred4,
        )


# Compatibility with the original testing script
Network = WaveletBoundaryCODNet


# ============================================================
# Quick Test
# ============================================================

if __name__ == "__main__":

    model = WaveletBoundaryCODNet(
        channels=48,
        backbone_checkpoint=None,
    )

    model.eval()

    x = torch.randn(
        1,
        3,
        384,
        384,
    )

    with torch.no_grad():
        outputs = model(x)

    for index, output in enumerate(outputs):
        print(
            "Output {}: {}".format(
                index,
                tuple(output.shape),
            )
        )
