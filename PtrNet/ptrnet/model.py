import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, C, T, V)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # (B, T, V, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, T, V)
        return self.act(x)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0))
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, C, T, V)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # (B, T, V, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return self.act(x)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_kernel=9):
        super().__init__()
        self.spatial = SpatialGraphConv(in_channels, out_channels)
        self.temporal = TemporalConv(out_channels, out_channels, kernel_size=temporal_kernel)
        self.residual = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        res = self.residual(x)
        x = self.temporal(self.spatial(x)) + res
        return F.gelu(x)


class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.scale = in_dim ** -0.5

    def forward(self, x):
        # x: (B, T, C)
        Q, K, V = self.query(x), self.key(x), self.value(x)
        attn = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) * self.scale, dim=-1)
        out = torch.bmm(attn, V)
        return out.mean(dim=1)  # (B, C)


class PtrNet(nn.Module):
    def __init__(self, num_joints=21, in_channels=3, lstm_hidden=128, gcn_dims=(64, 128, 256)):
        super().__init__()
        self.num_joints = num_joints
        self.input_dim = num_joints * in_channels

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.temporal_norm = nn.LayerNorm(lstm_hidden * 2)

        self.reshape_fc = nn.Linear(lstm_hidden * 2, self.num_joints * in_channels)

        self.stgcn_blocks = nn.ModuleList([
            STGCNBlock(3, gcn_dims[0]),
            STGCNBlock(gcn_dims[0], gcn_dims[1]),
            STGCNBlock(gcn_dims[1], gcn_dims[2])
        ])

        self.temporal_attn = AttentionPooling(gcn_dims[2])

        self.regressor = nn.Sequential(
            nn.Linear(gcn_dims[2], 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask=None):
        """
        x: (B, T, 63) body-pose trajectory features
        mask: (B, T) bool mask of valid timesteps
        Returns a score in normalized [0, 1]; multiply by 5 to recover the
        original 0-5 human-likeness scale.
        """
        B, T, C = x.shape
        device = x.device

        if mask is not None:
            lengths = mask.sum(dim=1).long()
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=T)
        else:
            out, _ = self.lstm(x)

        out = self.temporal_norm(out)

        graph_in = self.reshape_fc(out)  # (B, T, 63)
        graph_in = graph_in.view(B, T, self.num_joints, 3).permute(0, 3, 1, 2).contiguous()

        gcn_out = graph_in
        for block in self.stgcn_blocks:
            gcn_out = block(gcn_out)

        pooled = gcn_out.mean(dim=-1).permute(0, 2, 1)  # (B, T, C)
        pooled = self.temporal_attn(pooled)  # (B, C)

        score = self.regressor(pooled)
        return score.squeeze(1)
