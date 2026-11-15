import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

__all__ = ['HybridOutbreakModel']


class HybridOutbreakModel(nn.Module):
    """Hybrid LSTM + Transformer outbreak risk model.

    Architecture (approved):
      input: [B, 21, 25] float32
      -> 2-layer bidirectional LSTM, hidden_size=128
         -> last-time-step pooling -> [B, 256]
      -> Transformer branch: Linear(25->256), learned positional embedding (21),
         2 TransformerEncoder layers, 4 heads, ff=256, dropout=0.1,
         batch_first=True, last-time-step pooling -> [B, 256]
      -> concat -> [B, 512]
      -> MLP 512->128->1 with dropout 0.1
      -> logits [B]
      predict_proba() -> sigmoid([B]) in (0,1)
    """

    def __init__(
        self,
        input_dim: int = 25,
        seq_len: int = 21,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_hidden: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_hidden = ff_hidden
        self.dropout_p = dropout

        # --- LSTM branch ---
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # LSTM outputs: [B, seq_len, hidden_size*2] (bidirectional)
        self.lstm_out_dim = hidden_size * 2

        # --- Transformer branch ---
        self.transformer_input = nn.Linear(input_dim, ff_hidden)

        self.pos_embedding = nn.Parameter(torch.empty(seq_len, ff_hidden))
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

        encoder_layer = TransformerEncoderLayer(
            d_model=ff_hidden,
            nhead=num_heads,
            dim_feedforward=ff_hidden,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.transformer_out_dim = ff_hidden

        # --- Fusion + classifier ---
        fused_dim = self.lstm_out_dim + self.transformer_out_dim

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, seq_len, input_dim] float32 -> logits: [B] float32"""
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [B, seq_len, input_dim], got {x.dim()}D")
        if x.size(1) != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {x.size(1)}")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.size(-1)}")

        # LSTM branch [B, seq_len, input_dim] -> [B, seq_len, lstm_out_dim]
        lstm_out, _ = self.lstm(x)
        # last-time-step pooling
        lstm_pooled = lstm_out[:, -1, :]  # [B, lstm_out_dim]

        # Transformer branch
        t = self.transformer_input(x)                      # [B, T, ff_hidden]
        t = t + self.pos_embedding[: self.seq_len, :]          # broadcast [B, T, ff_hidden]
        t = self.transformer_encoder(t)                    # [B, T, ff_hidden]
        transformer_pooled = t[:, -1, :]                   # [B, ff_hidden]

        # Fusion
        fused = torch.cat([lstm_pooled, transformer_pooled], dim=-1)  # [B, fused_dim]

        logits = self.classifier(fused).squeeze(-1)        # [B]
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, seq_len, input_dim] -> probabilities [B] in (0,1)"""
        logits = self.forward(x)
        return torch.sigmoid(logits)
