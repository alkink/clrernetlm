from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointEmbedding(nn.Module):
    """Embedding for LaneLM keypoint tokens.

    Follows the LaneLM formulation:
      - x token in [0, nbins_x)
      - y token / time-step in [0, T], where T is used as padding (no lane).
    We embed x, y and the position index, then sum them.
    """

    def __init__(
        self,
        nbins_x: int,
        max_y_tokens: int,
        embed_dim: int,
        max_len: int,
    ) -> None:
        super().__init__()
        self.nbins_x = nbins_x
        self.max_y_tokens = max_y_tokens
        self.embed_dim = embed_dim
        self.max_len = max_len

        self.x_embedding = nn.Embedding(nbins_x, embed_dim)
        self.y_embedding = nn.Embedding(max_y_tokens, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x_tokens: torch.Tensor, y_tokens: torch.Tensor) -> torch.Tensor:
        """Embed (x_tokens, y_tokens) into a sequence of vectors.

        Args:
            x_tokens: LongTensor of shape (B, T)
            y_tokens: LongTensor of shape (B, T)

        Returns:
            Tensor of shape (B, T, D)
        """
        if x_tokens.dim() != 2 or y_tokens.dim() != 2:
            raise ValueError("x_tokens and y_tokens must have shape (B, T).")

        batch_size, seq_len = x_tokens.shape
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len={self.max_len}."
            )

        device = x_tokens.device
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        x_emb = self.x_embedding(x_tokens)
        y_emb = self.y_embedding(y_tokens)
        pos_emb = self.pos_embedding(pos_ids)

        return x_emb + y_emb + pos_emb


class LaneLMDecoderLayer(nn.Module):
    """Single Transformer decoder layer with self- and cross-attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
        - tgt: (B, T, D) token embeddings
        - memory: (B, N, D) visual tokens
        - tgt_mask: (T, T) causal mask (True for masked positions)
        - memory_key_padding_mask: (B, N) mask for visual tokens
        """
        # Self-attention (causal)
        residual = tgt
        attn_out, _ = self.self_attn(
            tgt,
            tgt,
            tgt,
            attn_mask=tgt_mask,
            need_weights=False,
        )
        tgt = self.norm1(residual + self.dropout(attn_out))

        # Cross-attention (image → language)
        residual = tgt
        attn_out, _ = self.cross_attn(
            tgt,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        tgt = self.norm2(residual + self.dropout(attn_out))

        # Feed-forward
        residual = tgt
        ffn_out = self.ffn(tgt)
        tgt = self.norm3(residual + self.dropout(ffn_out))
        return tgt


class LaneLMDecoder(nn.Module):
    """Stack of decoder layers operating on keypoint token embeddings."""

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                LaneLMDecoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    @staticmethod
    def _generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        # True values in the upper triangle (future positions) will be masked
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
        return mask

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
        - tgt: (B, T, D)
        - memory: (B, N, D)
        - memory_key_padding_mask: (B, N) or None
        """
        batch_size, seq_len, _ = tgt.shape
        device = tgt.device
        tgt_mask = self._generate_causal_mask(seq_len, device)

        out = tgt
        for layer in self.layers:
            out = layer(
                out,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        out = self.norm(out)
        return out


class LaneLMHead(nn.Module):
    """Projection head that predicts x and y tokens from decoder outputs."""

    def __init__(
        self,
        embed_dim: int,
        nbins_x: int,
        max_y_tokens: int,
    ) -> None:
        super().__init__()
        self.proj_x = nn.Linear(embed_dim, nbins_x)
        self.proj_y = nn.Linear(embed_dim, max_y_tokens)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
        - hidden_states: (B, T, D)

        Returns:
        - logits_x: (B, T, nbins_x)
        - logits_y: (B, T, max_y_tokens)
        """
        logits_x = self.proj_x(hidden_states)
        logits_y = self.proj_y(hidden_states)
        return logits_x, logits_y


class LaneLMModel(nn.Module):
    """LaneLM-style model that consumes visual features and lane tokens.

    This class intentionally does not depend on a specific backbone implementation.
    It expects precomputed visual features (e.g. from CLRerNet's backbone + neck)
    and provides the LaneLM decoding head on top.
    """

    def __init__(
        self,
        nbins_x: int,
        max_y_tokens: int,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 512,
        max_seq_len: int = 64,
        dropout: float = 0.1,
        visual_in_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.nbins_x = nbins_x
        self.max_y_tokens = max_y_tokens
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Optional projection from backbone feature dimension to embed_dim.
        # If visual_in_dim is None or equals embed_dim, the projection becomes identity.
        self.visual_in_dim = visual_in_dim or embed_dim
        if self.visual_in_dim != embed_dim:
            self.visual_proj = nn.Linear(self.visual_in_dim, embed_dim)
        else:
            self.visual_proj = None

        self.keypoint_embed = KeypointEmbedding(
            nbins_x=nbins_x,
            max_y_tokens=max_y_tokens,
            embed_dim=embed_dim,
            max_len=max_seq_len,
        )
        self.decoder = LaneLMDecoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.head = LaneLMHead(
            embed_dim=embed_dim,
            nbins_x=nbins_x,
            max_y_tokens=max_y_tokens,
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        x_tokens: torch.Tensor,
        y_tokens: torch.Tensor,
        visual_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            visual_tokens: (B, N, D_v) visual feature tokens. If D_v != embed_dim,
                it should be projected beforehand by the caller.
            x_tokens: (B, T) discrete x tokens.
            y_tokens: (B, T) discrete y / step tokens.
            visual_padding_mask: optional (B, N) mask for visual tokens.

        Returns:
            logits_x: (B, T, nbins_x)
            logits_y: (B, T, max_y_tokens)
        """
        # Project visual tokens to the model's embedding dimension if necessary.
        if self.visual_proj is not None:
            visual_tokens = self.visual_proj(visual_tokens)

        keypoint_emb = self.keypoint_embed(x_tokens, y_tokens)
        hidden = self.decoder(
            tgt=keypoint_emb,
            memory=visual_tokens,
            memory_key_padding_mask=visual_padding_mask,
        )
        logits_x, logits_y = self.head(hidden)
        return logits_x, logits_y
