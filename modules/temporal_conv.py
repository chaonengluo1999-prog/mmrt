"""
MMRT Module 2: Multi-scale Temporal Encoder
Multi-scale 1D temporal convolution for sign language sequence modeling.
"""
import copy
import torch
import torch.nn as nn


class MultiScaleDilatedConv(nn.Module):
    """Multi-branch dilated temporal convolution."""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilations=(1, 2, 3, 4)):
        super().__init__()
        self.num_branches = 4
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels // self.num_branches,
                          kernel_size=kernel_size, dilation=d, padding=d),
                nn.BatchNorm1d(out_channels // self.num_branches)
            )
            for d in dilations
        ])

    def forward(self, x):
        return torch.cat([b(x) for b in self.branches], dim=1)


class MultiScaleTemporalEncoder(nn.Module):
    """
    MMRT Module 2: Multi-scale temporal convolution encoder.
    Processes frame-level features into sequence-level representations.
    """

    def __init__(self, input_size, hidden_size, conv_type=2, use_bn=False, num_classes=-1):
        super().__init__()
        self.use_bn = use_bn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.conv_type = conv_type

        kernel_map = {
            0: ['K3'],
            1: ['K5', 'P2'],
            2: ['K5', 'P2', 'K5', 'P2'],
            3: ['K5', 'K5', 'P2'],
            4: ['K5', 'K5'],
            5: ['K5', 'P2', 'K5'],
            6: ['P2', 'K5', 'K5'],
            7: ['P2', 'K5', 'P2', 'K5'],
            8: ['P2', 'P2', 'K5', 'K5'],
        }
        self.kernel_size = kernel_map[conv_type]

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = (self.input_size if layer_idx == 0
                        or (self.conv_type == 6 and layer_idx == 1)
                        or (self.conv_type == 7 and layer_idx == 1)
                        or (self.conv_type == 8 and layer_idx == 2)
                        else self.hidden_size)
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]),
                                         stride=1, padding=0))
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)

        if self.num_classes != -1:
            self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def update_lgt(self, lgt):
        feat_len = copy.deepcopy(lgt)
        for ks in self.kernel_size:
            if ks[0] == 'P':
                feat_len = torch.div(feat_len, 2)
            else:
                feat_len -= int(ks[1]) - 1
        return feat_len

    def forward(self, frame_feat, lgt):
        visual_feat = self.temporal_conv(frame_feat)
        lgt = self.update_lgt(lgt)
        logits = (None if self.num_classes == -1
                  else self.fc(visual_feat.transpose(1, 2)).transpose(1, 2))
        return {
            "visual_feat": visual_feat.permute(2, 0, 1),
            "conv_logits": logits.permute(2, 0, 1),
            "feat_len": lgt,
        }


# Alias for state_dict compatibility
TemporalConv = MultiScaleTemporalEncoder
