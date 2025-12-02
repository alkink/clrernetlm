from typing import Iterable, List, Optional, Sequence, Tuple

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
        x_embedding_scale: float = 1.0,  # V5: Scale factor for X embedding (reduce dependency on past X)
        lane_embedding_boost: float = 10.0,  # V5: Boost factor for lane embedding (emphasize visual info)
    ) -> None:
        super().__init__()
        self.nbins_x = nbins_x
        self.max_y_tokens = max_y_tokens
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.x_embedding_scale = x_embedding_scale  # V5: Default 1.0, V5 uses 0.3
        self.lane_embedding_boost = lane_embedding_boost  # V5: Default 10.0, V5 uses 15.0

        self.x_embedding = nn.Embedding(nbins_x, embed_dim)
        self.y_embedding = nn.Embedding(max_y_tokens, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        # Lane ID Embedding (for Multi-Lane One-to-Many handling)
        self.lane_embedding = nn.Embedding(8, embed_dim) # Support up to 8 lanes

    def forward(self, x_tokens: torch.Tensor, y_tokens: torch.Tensor, lane_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Embed (x_tokens, y_tokens) into a sequence of vectors.

        Args:
            x_tokens: LongTensor of shape (B, T)
            y_tokens: LongTensor of shape (B, T)
            lane_indices: LongTensor of shape (B,) or (B, T) - Optional lane ID.

        Returns:
            Tensor of shape (B, T, D)
        """
        if x_tokens.dim() != 2 or y_tokens.dim() != 2:
            raise ValueError("x_tokens and y_tokens must have shape (B, T).")

        batch_size, seq_len = x_tokens.shape

        # Validate sequence length
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len={self.max_len}."
            )

        # Validate token ranges to prevent embedding lookup errors
        if seq_len > 0:
            x_max = x_tokens.max().item()
            y_max = y_tokens.max().item()

            if x_max >= self.nbins_x:
                raise ValueError(
                    f"x_tokens contain value {x_max} >= nbins_x={self.nbins_x}. "
                    f"This indicates a configuration mismatch between training and inference."
                )
            if y_max >= self.max_y_tokens:
                raise ValueError(
                    f"y_tokens contain value {y_max} >= max_y_tokens={self.max_y_tokens}. "
                    f"y_tokens should be in range [0, {self.max_y_tokens-1}]."
                )

        device = x_tokens.device
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # V5: Scale X embedding to reduce dependency on past X tokens
        x_emb = self.x_embedding(x_tokens) * self.x_embedding_scale
        y_emb = self.y_embedding(y_tokens)
        pos_emb = self.pos_embedding(pos_ids)

        out = x_emb + y_emb + pos_emb
        
        if lane_indices is not None:
            # If lane_indices is (B,), expand to (B, T)
            if lane_indices.dim() == 1:
                lane_indices = lane_indices.unsqueeze(1).expand(-1, seq_len)
            
            lane_emb = self.lane_embedding(lane_indices)
            # V5: Boost Lane Embedding to emphasize visual information
            # Original: 10.0, V5: 15.0 (stronger signal for which lane to predict)
            out = out + (lane_emb * self.lane_embedding_boost)

        return out


class VisualTokenEncoder(nn.Module):
    """Encode multi-scale CNN/FPN feature maps into visual tokens.

    This follows the LaneLM formulation (Eq. 7-8 in the paper):
      - use a pyramid feature extractor f to obtain {F0, F1, F2}
      - split each Fi into patches, linearly embed them and add
        positional and level embeddings.

    In practice we assume Fi are the FPN outputs from CLRerNetFPN:
      Fi: (B, C_i, H_i, W_i)
    and flatten each spatial location as a "patch".
    
    Key improvement: Uses 2D sinusoidal positional embeddings to preserve
    spatial structure, which is critical for lane detection.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        embed_dim: int,
        use_2d_pe: bool = True,  # NEW: Enable 2D positional embedding
        use_adaptive_pooling: bool = True,  # V5: Adaptive spatial pooling to reduce tokens
        target_spatial_size: Optional[Tuple[int, int]] = None,  # V5: Target (H, W) for pooling
    ) -> None:
        super().__init__()
        self.in_channels = list(in_channels)
        self.embed_dim = embed_dim
        self.use_2d_pe = use_2d_pe
        self.use_adaptive_pooling = use_adaptive_pooling
        self.target_spatial_size = target_spatial_size

        # Per-level LayerNorm + linear projection to embed_dim
        # LayerNorm is critical because CLRerNet FPN outputs have large variance!
        self.norm_per_level = nn.ModuleList(
            [nn.LayerNorm(c) for c in self.in_channels]
        )
        self.proj_per_level = nn.ModuleList(
            [nn.Linear(c, embed_dim) for c in self.in_channels]
        )

        # V5: Adaptive pooling per level (if enabled)
        if use_adaptive_pooling and target_spatial_size is not None:
            self.adaptive_pool_per_level = nn.ModuleList([
                nn.AdaptiveAvgPool2d(target_spatial_size)
                for _ in self.in_channels
            ])
        else:
            self.adaptive_pool_per_level = None

        # Level embeddings (encode which FPN level a token comes from)
        self.level_embed = nn.Embedding(len(self.in_channels), embed_dim)

        # Positional embedding options
        if use_2d_pe:
            # 2D sinusoidal PE will be computed on-the-fly based on H, W
            # No learnable parameters needed for sinusoidal
            pass
        else:
            # Fallback: 1D learnable positional embedding (legacy)
            self.max_tokens = 8192
            self.pos_embedding = nn.Embedding(self.max_tokens, embed_dim)

    def _get_2d_sincos_pos_embed(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Generate 2D sinusoidal positional embeddings.
        
        This preserves spatial structure by encoding (x, y) positions separately
        and concatenating them, similar to ViT and DETR.
        
        V5 Enhancement: Stronger frequency bands and scaling for better spatial awareness.
        
        Returns: (H*W, embed_dim) tensor
        """
        embed_dim = self.embed_dim
        assert embed_dim % 2 == 0, "embed_dim must be divisible by 2 for 2D PE"
        
        half_dim = embed_dim // 2
        
        # Create position grids
        y_pos = torch.arange(H, device=device, dtype=torch.float32)
        x_pos = torch.arange(W, device=device, dtype=torch.float32)
        
        # Normalize to [0, 1]
        y_pos = y_pos / max(H - 1, 1)
        x_pos = x_pos / max(W - 1, 1)
        
        # Create meshgrid
        y_grid, x_grid = torch.meshgrid(y_pos, x_pos, indexing='ij')
        y_grid = y_grid.reshape(-1)  # (H*W,)
        x_grid = x_grid.reshape(-1)  # (H*W,)
        
        # V5: Stronger frequency bands - use more frequency components
        # Original: half_dim // 2 frequencies
        # Enhanced: Use more frequencies for better spatial resolution
        num_freqs = half_dim // 2
        freq_bands = torch.arange(num_freqs, device=device, dtype=torch.float32)
        # V5: Scale factor for stronger positional signal
        freq_scale = 2.0  # Increase frequency strength
        freq_bands = freq_scale / (10000 ** (2 * freq_bands / num_freqs))
        
        # Compute sin/cos embeddings for x and y
        x_embed = x_grid.unsqueeze(-1) * freq_bands.unsqueeze(0) * 3.14159  # (H*W, num_freqs)
        y_embed = y_grid.unsqueeze(-1) * freq_bands.unsqueeze(0) * 3.14159  # (H*W, num_freqs)
        
        # Interleave sin and cos
        x_pe = torch.stack([x_embed.sin(), x_embed.cos()], dim=-1).reshape(-1, num_freqs * 2)  # (H*W, num_freqs*2)
        y_pe = torch.stack([y_embed.sin(), y_embed.cos()], dim=-1).reshape(-1, num_freqs * 2)  # (H*W, num_freqs*2)
        
        # V5: If half_dim > num_freqs*2, pad with zeros or repeat
        if half_dim > num_freqs * 2:
            # Repeat to fill half_dim
            repeat_factor = (half_dim + num_freqs * 2 - 1) // (num_freqs * 2)
            x_pe = x_pe.repeat(1, repeat_factor)[:, :half_dim]
            y_pe = y_pe.repeat(1, repeat_factor)[:, :half_dim]
        elif half_dim < num_freqs * 2:
            # Truncate if needed
            x_pe = x_pe[:, :half_dim]
            y_pe = y_pe[:, :half_dim]
        
        # Concatenate x and y embeddings
        pos_embed = torch.cat([x_pe, y_pe], dim=-1)  # (H*W, embed_dim)
        
        # V5: Scale positional embedding for stronger signal
        pos_scale = 1.5  # Boost positional signal strength
        pos_embed = pos_embed * pos_scale
        
        return pos_embed

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        """Encode FPN feature maps into a single token sequence.

        Args:
            feats: sequence of tensors [F0, F1, F2], each of shape (B, C_i, H_i, W_i).

        Returns:
            visual_tokens: (B, N_total, D) where N_total=sum_i H_i*W_i.
        """
        if len(feats) != len(self.in_channels):
            raise ValueError(
                f"Expected {len(self.in_channels)} feature maps, "
                f"got {len(feats)}."
            )

        batch_size = feats[0].shape[0]
        device = feats[0].device

        level_tokens: List[torch.Tensor] = []
        for lvl, (feat, norm, proj) in enumerate(zip(feats, self.norm_per_level, self.proj_per_level)):
            B, C, H, W = feat.shape
            if C != self.in_channels[lvl]:
                raise ValueError(
                    f"Unexpected channels at level {lvl}: "
                    f"got {C}, expected {self.in_channels[lvl]}."
                )

            # V5: Apply adaptive pooling if enabled
            if self.adaptive_pool_per_level is not None:
                pool = self.adaptive_pool_per_level[lvl]
                feat = pool(feat)  # (B, C, H_target, W_target)
                H, W = feat.shape[2], feat.shape[3]

            # (B, C, H, W) -> (B, H*W, C)
            x = feat.view(B, C, H * W).permute(0, 2, 1).contiguous()
            # LayerNorm to stabilize the large-variance FPN outputs
            x = norm(x)  # (B, H*W, C) normalized
            # Linear projection to embed_dim
            x = proj(x)  # (B, H*W, D)

            # Add 2D positional embedding for THIS level
            if self.use_2d_pe:
                pos_emb = self._get_2d_sincos_pos_embed(H, W, device)  # (H*W, D)
                x = x + pos_emb.unsqueeze(0)  # Broadcast over batch

            # Add level embedding
            lvl_emb = self.level_embed.weight[lvl].view(1, 1, -1)  # (1,1,D)
            x = x + lvl_emb
            level_tokens.append(x)

        # Concatenate levels along token dimension
        visual_tokens = torch.cat(level_tokens, dim=1)  # (B, N_total, D)
        
        # If using legacy 1D PE (not recommended), apply it here
        if not self.use_2d_pe:
            _, n_tokens, _ = visual_tokens.shape
            if n_tokens > self.max_tokens:
                raise ValueError(
                    f"Number of visual tokens {n_tokens} exceeds "
                    f"max_tokens={self.max_tokens}."
                )
            pos_ids = torch.arange(n_tokens, device=device).unsqueeze(0).expand(
                batch_size, -1
            )
            pos_emb = self.pos_embedding(pos_ids)
            visual_tokens = visual_tokens + pos_emb

        return visual_tokens


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

        # V5: Self-attention with higher dropout to reduce past X dependency
        self.self_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=max(dropout, 0.2),  # V5: Minimum 0.2 dropout for self-attention
            batch_first=True,
        )
        # V5: Cross-attention with more heads for stronger visual information
        cross_attn_heads = num_heads * 2 if num_heads < 16 else num_heads  # Double heads if < 16
        self.cross_attn = nn.MultiheadAttention(
            embed_dim,
            cross_attn_heads,
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
        
        V5: Visual-First Decoder - Cross-attention FIRST, then self-attention
        This makes visual information primary and reduces dependency on past X tokens.
        """
        # V5: Cross-attention FIRST (visual information is primary)
        residual = tgt
        attn_out, attn_weights = self.cross_attn(
            tgt,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,  # Keep weights for debugging
        )
        # V5: Visual-Query Fusion - add cross-attention output to original query
        tgt = self.norm1(residual + self.dropout(attn_out))

        # V5: Self-attention SECOND (weakened, with higher dropout)
        residual = tgt
        attn_out, _ = self.self_attn(
            tgt,
            tgt,
            tgt,
            attn_mask=tgt_mask,
            need_weights=False,
        )
        # V5: Higher dropout for self-attention to reduce past X dependency
        tgt = self.norm2(residual + self.dropout(attn_out * 0.8))  # Scale down self-attention
        
        # Store attention weights for debugging (if needed)
        # attn_weights shape: (B, num_heads, T, N) where N is num visual tokens

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
        # Input validation to prevent CUDA device-side assert
        if not isinstance(seq_len, int) or seq_len <= 0:
            raise ValueError(f"Invalid seq_len: {seq_len}. Must be a positive integer.")
        if seq_len > 1024:  # Reasonable upper limit to prevent memory issues
            raise ValueError(f"seq_len too large: {seq_len}. Maximum allowed is 1024.")

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
        # Validate input tensor shapes
        if tgt.dim() != 3:
            raise ValueError(f"tgt must be 3D tensor (B, T, D), got shape {tgt.shape}")

        batch_size, seq_len, embed_dim = tgt.shape

        # Early validation to prevent CUDA device-side assert
        if seq_len <= 0:
            raise ValueError(f"Invalid sequence length: {seq_len}. tgt.shape: {tgt.shape}")
        if seq_len > 1024:  # Reasonable upper limit
            raise ValueError(f"Sequence length too large: {seq_len}. Maximum allowed is 1024.")

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
        visual_in_channels: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()

        # Validation to prevent configuration mismatches
        if nbins_x <= 0:
            raise ValueError(f"nbins_x must be positive, got {nbins_x}")
        if nbins_x < 300:
            print(f"WARNING: nbins_x={nbins_x} < 300. If using BOS tokens [296,297,298,299], this may cause index out of bounds errors.")
            print("Consider setting nbins_x=300 to include BOS tokens and relative token space.")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

        self.nbins_x = nbins_x
        self.max_y_tokens = max_y_tokens
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Visual encoder: either a simple linear projection from a single
        # feature map (legacy path, visual_in_dim) or a multi-scale token
        # encoder over a pyramid of feature maps (visual_in_channels).
        self.visual_encoder: Optional[VisualTokenEncoder]
        if visual_in_channels is not None:
            # Multi-level FPN features -> tokens
            # V5: Use adaptive pooling for P5-only to reduce tokens (250 -> 65)
            # For P5: (10, 25) -> (5, 13) = 65 tokens
            use_adaptive_pooling = len(visual_in_channels) == 1  # Only for single-level (P5-only)
            target_spatial_size = (5, 13) if use_adaptive_pooling else None
            
            self.visual_encoder = VisualTokenEncoder(
                in_channels=visual_in_channels,
                embed_dim=embed_dim,
                use_2d_pe=True,
                use_adaptive_pooling=use_adaptive_pooling,
                target_spatial_size=target_spatial_size,
            )
            self.visual_proj = None
        else:
            self.visual_encoder = None
            # Optional projection from backbone feature dimension to embed_dim.
            # If visual_in_dim is None or equals embed_dim, the projection becomes identity.
            self.visual_in_dim = visual_in_dim or embed_dim
            if self.visual_in_dim != embed_dim:
                self.visual_proj = nn.Linear(self.visual_in_dim, embed_dim)
            else:
                self.visual_proj = None

        # V5: Keypoint Embedding with reduced X dependency and boosted lane signal
        self.keypoint_embed = KeypointEmbedding(
            nbins_x=nbins_x,
            max_y_tokens=max_y_tokens,
            embed_dim=embed_dim,
            max_len=max_seq_len,
            x_embedding_scale=0.3,  # V5: Reduce past X dependency (1.0 -> 0.3)
            lane_embedding_boost=15.0,  # V5: Boost lane signal (10.0 -> 15.0)
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
        lane_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            visual_tokens:
                - If visual_encoder is None:
                    (B, N, D_v) visual feature tokens. If D_v != embed_dim,
                    it will be projected to embed_dim.
                - If visual_encoder is not None:
                    sequence of FPN feature maps is expected instead and this
                    argument will be ignored (see note below).

            x_tokens: (B, T) discrete x tokens.
            y_tokens: (B, T) discrete y / step tokens.
            visual_padding_mask: optional (B, N) mask for visual tokens.
            lane_indices: optional (B,) lane IDs for multi-lane prediction.

        Returns:
            logits_x: (B, T, nbins_x)
            logits_y: (B, T, max_y_tokens)
        Note:
            - If ``visual_in_channels`` was provided at construction time,
              callers are expected to pass the output of
              :meth:`encode_visual_tokens` as ``visual_tokens``. In that case
              no further projection is applied here.
            - If only ``visual_in_dim`` was provided, ``visual_tokens`` is
              projected to ``embed_dim`` when necessary.
        """
        # Early validation to catch BOS token vocabulary mismatches
        if x_tokens.dim() != 2 or y_tokens.dim() != 2:
            raise ValueError(f"x_tokens and y_tokens must be 2D tensors, got shapes {x_tokens.shape}, {y_tokens.shape}")

        # Validate token ranges before embedding to prevent index out of bounds
        x_max = x_tokens.max().item()
        y_max = y_tokens.max().item()

        if x_max >= self.nbins_x:
            raise ValueError(
                f"x_tokens contain value {x_max} >= nbins_x={self.nbins_x}. "
                f"This usually happens when:\n"
                f"  1. Training used nbins_x=300 with BOS tokens [296,297,298,299]\n"
                f"  2. Inference uses nbins_x={self.nbins_x}\n"
                f"  3. Solution: Set nbins_x=300 in test config"
            )
        if y_max >= self.max_y_tokens:
            raise ValueError(
                f"y_tokens contain value {y_max} >= max_y_tokens={self.max_y_tokens}"
            )

        # Encode / project visual tokens to the model's embedding dimension.
        if self.visual_encoder is not None:
            # visual_tokens are assumed to be already encoded to (B, N, D).
            pass
        elif self.visual_proj is not None:
            visual_tokens = self.visual_proj(visual_tokens)

        keypoint_emb = self.keypoint_embed(x_tokens, y_tokens, lane_indices)
        hidden = self.decoder(
            tgt=keypoint_emb,
            memory=visual_tokens,
            memory_key_padding_mask=visual_padding_mask,
        )
        logits_x, logits_y = self.head(hidden)
        return logits_x, logits_y

    def encode_visual_tokens(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        """Encode multi-level FPN features into visual tokens.

        This is the preferred entry point when LaneLMModel was constructed
        with ``visual_in_channels``. It wraps ``VisualTokenEncoder``.
        """
        if self.visual_encoder is None:
            raise RuntimeError(
                "encode_visual_tokens() was called but LaneLMModel was "
                "constructed without visual_in_channels."
            )
        return self.visual_encoder(feats)
