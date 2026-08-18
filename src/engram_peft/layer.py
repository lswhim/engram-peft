# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none, reportUnknownArgumentType=none
import os
from typing import Any, cast, final, override

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int64

from engram_peft.compression import CompressedTokenizer
from engram_peft.config import EngramConfig
from engram_peft.hashing import FixedNgramHashMapping, NgramHashMapping
from engram_peft.rq_hashing import RQNgramMapping
from engram_peft.types import jaxtyped
from engram_peft.utils import safe_from_numpy


@final
class ShortConv(nn.Module):
    """
    ShortConv module as described in the Engram paper.

    Y = SiLU( Conv1D( RMSNorm(Ṽ) ) ) + Ṽ
    Applies independent RMSNorm per branch, depthwise causal convolution,
    and optional SiLU activation, followed by a residual connection.
    """

    hidden_size: int
    kernel_size: int
    dilation: int
    hc_mult: int
    activation: bool
    norms: nn.ModuleList
    conv: nn.Conv1d

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 4,
        dilation: int = 1,
        norm_eps: float = 1e-5,
        hc_mult: int = 4,
        activation: bool = True,
        zero_init: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.hc_mult = hc_mult
        self.activation = activation

        # Step 1: Independent RMSNorm for each branch
        self.norms = nn.ModuleList(
            [nn.RMSNorm(hidden_size, eps=norm_eps) for _ in range(hc_mult)]
        )

        # Depthwise convolution (groups=total_channels)
        total_channels = hc_mult * hidden_size
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            groups=total_channels,
            bias=False,
        )

        # Weight/bias initialization
        if zero_init:
            nn.init.zeros_(self.conv.weight)
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)
        else:
            # Use normal initialization if zero_init is False
            nn.init.normal_(self.conv.weight, std=0.02)
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)

    @jaxtyped
    @override
    def forward(
        self, x: Float[torch.Tensor, "batch seq hc_mult hidden_dim"]
    ) -> Float[torch.Tensor, "batch seq hc_mult hidden_dim"]:
        """
        Forward pass for ShortConv.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hc_mult, hidden_size]

        Returns:
            torch.Tensor: Output tensor of same shape, calculated as SiLU(Conv(Norm(x))) + x
        """
        batch_size, seq_len, hc_mult, hidden_size = x.shape

        # Step 1: Independent RMSNorm per branch
        normed_branches: list[torch.Tensor] = []
        for i in range(hc_mult):
            normed_branches.append(self.norms[i](x[:, :, i, :]))
        x_norm = torch.stack(normed_branches, dim=2)

        # Step 2: Reshape for Conv1D -> [batch_size, total_channels, seq_len]
        # x_norm: [B, L, M, D] -> [B, M*D, L]
        x_conv_in = x_norm.permute(0, 2, 3, 1).reshape(
            batch_size, hc_mult * hidden_size, seq_len
        )

        # Step 4: Causal padding (shift sequence right so conv output corresponds to current pos)
        pad_len = (self.kernel_size - 1) * self.dilation
        if pad_len > 0:
            x_padded = F.pad(x_conv_in, (pad_len, 0))
        else:
            x_padded = x_conv_in

        # Step 3: Depthwise convolution
        conv_out = self.conv(x_padded)

        # Step 5: SiLU activation (in-place saves one intermediate tensor)
        if self.activation:
            conv_out = F.silu(conv_out, inplace=True)

        # Step 6: Convert back to [batch_size, seq_len, hc_mult, hidden_size]
        out = (
            conv_out.view(batch_size, hc_mult, hidden_size, seq_len)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        # Matches the formula: Y = SiLU(Conv(Norm(V))) + V
        return (out + x).to(x.dtype)


@final
class ContextAwareGating(nn.Module):
    """
    Context-Aware Gating module as described in Section 2.3 and 2.4 of the Engram paper.

    This module computes a gating signal based on the context (h_t) and the retrieved
    Engram embeddings (e_t), and applies it to the value projection of the embeddings.
    """

    config: EngramConfig
    engram_hidden_size: int
    hidden_size: int
    hc_mult: int
    w_v: nn.Linear
    w_k: nn.ModuleList
    norm_h: nn.ModuleList
    norm_k: nn.ModuleList
    last_gate: torch.Tensor | None
    last_entropy: float
    gating_entropy: torch.Tensor

    def __init__(
        self,
        config: EngramConfig,
        engram_hidden_size: int,
        hidden_size: int,
        hc_mult: int = 4,
        zero_init: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.engram_hidden_size = engram_hidden_size
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult

        # Step 1: Shared Value projection
        self.w_v = nn.Linear(engram_hidden_size, hidden_size, bias=False)
        if zero_init:
            nn.init.zeros_(self.w_v.weight)

        # Step 2: Branch-specific Key projection W_K^(m)
        self.w_k = nn.ModuleList(
            [
                nn.Linear(engram_hidden_size, hidden_size, bias=False)
                for _ in range(hc_mult)
            ]
        )

        # Step 3: Independent RMSNorm
        self.norm_h = nn.ModuleList([nn.RMSNorm(hidden_size) for _ in range(hc_mult)])
        self.norm_k = nn.ModuleList([nn.RMSNorm(hidden_size) for _ in range(hc_mult)])
        self.last_gate = None
        self.last_entropy = 0.0  # Default to zero

    @jaxtyped
    @override
    def forward(
        self,
        embeddings: Float[torch.Tensor, "batch seq engram_hidden"],
        hidden_states: Float[torch.Tensor, "batch seq hc_mult hidden_dim"],
    ) -> Float[torch.Tensor, "batch seq hc_mult hidden_dim"]:
        """
        Forward pass of the ContextAwareGating module.

        Args:
            embeddings: [batch_size, seq_len, engram_hidden_size]
            hidden_states: [batch_size, seq_len, hc_mult, hidden_size]

        Returns:
            torch.Tensor: gated_value of shape [batch_size, seq_len, hc_mult, hidden_size]
        """
        # Step 1: Shared Value projection
        value = self.w_v(embeddings)  # [B, L, D]

        # Compute gate scores per branch and stack small intermediates
        # (only stores M * [B, L, 1] in the list, avoids stacking full [B, L, D] tensors)
        gate_scores: list[torch.Tensor] = []
        for m in range(self.hc_mult):
            key_m = self.w_k[m](embeddings)
            normed_key = self.norm_k[m](key_m)
            normed_query = self.norm_h[m](hidden_states[:, :, m, :])
            gate_scores.append((normed_key * normed_query).sum(dim=-1, keepdim=True))

        gate = torch.stack(gate_scores, dim=2)  # [B, L, M, 1]
        gate = gate / (self.hidden_size**0.5)

        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gate = gate.sigmoid()  # [B, L, M, 1]

        # Store for visualization
        self.last_gate = gate.detach()

        # Calculate Entropy
        # p=gate, 1-p=(1-gate)
        p = gate.clamp(1e-6, 1 - 1e-6)
        # We always compute the tensor version for loss regularization support
        self.gating_entropy = -(p * p.log() + (1 - p) * (1 - p).log()).mean()

        if self.config.enable_telemetry:
            with torch.no_grad():
                self.last_entropy = self.gating_entropy.item()

        # Step 5: Gating modulation
        gated_value = gate * value.unsqueeze(
            2
        )  # [B, L, M, 1] * [B, L, 1, D] -> [B, L, M, D]

        return gated_value


@final
class HeadFactorizedGating(nn.Module):
    """Parameter-matched per-head alternative to flattened Engram gating.

    ``w_v`` and every ``w_k`` have exactly the same shapes as the original
    :class:`ContextAwareGating`.  Their input dimensions are interpreted as contiguous
    head blocks, so ``W @ concat(e_h) == sum_h W_h @ e_h``.  This exposes one route
    score per memory head without adding a second value network.
    """

    def __init__(
        self,
        config: EngramConfig,
        num_heads: int,
        embedding_dim_per_head: int,
        hidden_size: int,
        hc_mult: int = 4,
        zero_init: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.num_heads = num_heads
        self.embedding_dim_per_head = embedding_dim_per_head
        self.engram_hidden_size = num_heads * embedding_dim_per_head
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult

        self.w_v = nn.Linear(self.engram_hidden_size, hidden_size, bias=False)
        if zero_init:
            nn.init.zeros_(self.w_v.weight)
        self.w_k = nn.ModuleList(
            [
                nn.Linear(self.engram_hidden_size, hidden_size, bias=False)
                for _ in range(hc_mult)
            ]
        )
        self.norm_h = nn.ModuleList([nn.RMSNorm(hidden_size) for _ in range(hc_mult)])
        self.norm_k = nn.ModuleList([nn.RMSNorm(hidden_size) for _ in range(hc_mult)])
        self.last_gate: torch.Tensor | None = None
        self.last_route_logits: torch.Tensor | None = None
        self.last_entropy = 0.0
        self.gating_entropy = torch.tensor(0.0)
        self.forced_head_mask: torch.Tensor | None = None

    def set_forced_head_mask(self, mask: torch.Tensor | None) -> None:
        """Force a route for counterfactual training.

        Accepted shapes are ``[B,H]``, ``[B,L,H]``, or ``[B,L,M,H]`` and are
        broadcast to the computed route tensor.  A zero mask is the explicit null route.
        """
        self.forced_head_mask = mask

    def _expanded_mask(self, gate: torch.Tensor) -> torch.Tensor | None:
        mask = self.forced_head_mask
        if mask is None:
            return None
        mask = mask.to(device=gate.device, dtype=gate.dtype)
        if mask.dim() == 2:
            mask = mask[:, None, None, :]
        elif mask.dim() == 3:
            mask = mask[:, :, None, :]
        if mask.dim() != 4 or mask.shape[0] != gate.shape[0] or mask.shape[-1] != gate.shape[-1]:
            raise ValueError(
                "forced head mask must be [B,H], [B,L,H], or [B,L,M,H] "
                f"for gate shape {tuple(gate.shape)}; got {tuple(mask.shape)}"
            )
        return mask

    @override
    def forward(
        self,
        embeddings: Float[torch.Tensor, "batch seq heads head_dim"],
        hidden_states: Float[torch.Tensor, "batch seq hc_mult hidden_dim"],
        head_selection_scores: Float[torch.Tensor, "batch seq heads"] | None = None,
    ) -> Float[torch.Tensor, "batch seq hc_mult hidden_dim"]:
        if embeddings.shape[-2:] != (
            self.num_heads,
            self.embedding_dim_per_head,
        ):
            raise ValueError(
                "head-factorized embeddings have incompatible shape: "
                f"expected (..., {self.num_heads}, {self.embedding_dim_per_head}), "
                f"got {tuple(embeddings.shape)}"
            )

        value_blocks = self.w_v.weight.view(
            self.hidden_size, self.num_heads, self.embedding_dim_per_head
        )
        # [B,L,H,E] x [D,H,E] -> [B,L,H,D]
        values = torch.einsum("blhe,dhe->blhd", embeddings, value_blocks)

        branch_logits: list[torch.Tensor] = []
        for branch in range(self.hc_mult):
            key_blocks = self.w_k[branch].weight.view(
                self.hidden_size, self.num_heads, self.embedding_dim_per_head
            )
            keys = torch.einsum("blhe,dhe->blhd", embeddings, key_blocks)
            normed_keys = self.norm_k[branch](keys)
            query = self.norm_h[branch](hidden_states[:, :, branch, :])
            score = (normed_keys * query.unsqueeze(2)).sum(dim=-1)
            score = score / (self.hidden_size**0.5)
            score = score.abs().clamp_min(1e-6).sqrt() * score.sign()
            branch_logits.append(score)

        route_logits = torch.stack(branch_logits, dim=2)  # [B,L,M,H]
        gate = route_logits.sigmoid()
        dense_gate = gate

        forced_mask = self._expanded_mask(gate)
        if forced_mask is not None:
            gate = gate * forced_mask
        else:
            top_k = int(getattr(self.config, "head_router_top_k", 0) or 0)
            if 0 < top_k < self.num_heads:
                selection = str(
                    getattr(self.config, "head_router_selection", "context")
                )
                if selection == "context":
                    selection_logits = route_logits
                elif selection in {"specificity", "rq_snr", "rq_signal"}:
                    if head_selection_scores is None:
                        raise ValueError(
                            f"head_router_selection={selection!r} requires per-address scores"
                        )
                    if head_selection_scores.shape != route_logits.shape[:2] + (
                        self.num_heads,
                    ):
                        raise ValueError(
                            "head selection scores must be [B,L,H]; got "
                            f"{tuple(head_selection_scores.shape)}"
                        )
                    selection_logits = head_selection_scores.to(
                        device=route_logits.device, dtype=route_logits.dtype
                    )[:, :, None, :].expand_as(route_logits)
                else:
                    raise ValueError(
                        "head_router_selection must be 'context', 'specificity', 'rq_snr', or 'rq_signal', "
                        f"got {selection!r}"
                    )
                indices = selection_logits.topk(top_k, dim=-1).indices
                hard_mask = torch.zeros_like(gate).scatter_(-1, indices, 1.0)
                gate = gate * hard_mask
            if bool(getattr(self.config, "head_router_use_null", False)):
                threshold = float(
                    getattr(self.config, "head_router_null_threshold", 0.0) or 0.0
                )
                use_memory = route_logits.amax(dim=-1, keepdim=True) > threshold
                gate = gate * use_memory.to(gate.dtype)

        if bool(getattr(self.config, "head_router_preserve_mass", False)):
            # Sparse routing must change which heads contribute, not silently reduce
            # memory strength by k/H. Detaching the normalizer keeps gradients (and
            # writes) restricted to selected heads. Explicit null routes stay zero.
            selected_mass = gate.sum(dim=-1, keepdim=True)
            dense_mass = dense_gate.sum(dim=-1, keepdim=True)
            scale = torch.where(
                selected_mass > 0,
                dense_mass / selected_mass.clamp_min(1e-6),
                torch.zeros_like(selected_mass),
            ).detach()
            gate = gate * scale

        self.last_route_logits = route_logits
        self.last_gate = gate.detach()
        probability = route_logits.sigmoid().clamp(1e-6, 1 - 1e-6)
        self.gating_entropy = -(
            probability * probability.log()
            + (1 - probability) * (1 - probability).log()
        ).mean()
        if self.config.enable_telemetry:
            self.last_entropy = self.gating_entropy.item()

        # One independently gated contribution per head, summed without changing the
        # downstream ShortConv interface [B,L,M,D].
        return torch.einsum("blmh,blhd->blmd", gate, values)


@final
class MultiHeadEmbedding(nn.Module):
    """
    Concatenated embedding table for all hash heads across all N-gram sizes.
    Retrieves vectors from K independent virtual embedding tables using offset indices.
    """

    offsets: torch.Tensor
    embedding: nn.Embedding

    def __init__(
        self,
        primes: list[int],
        embedding_dim_per_head: int,
        sparse: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        if sparse and int(os.environ.get("WORLD_SIZE", "1")) > 1:
            sparse = False

        offsets = [0]
        for p in primes[:-1]:
            offsets.append(offsets[-1] + p)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

        total_capacity = sum(primes)
        self.embedding = nn.Embedding(
            total_capacity, embedding_dim_per_head, sparse=sparse
        )
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    @jaxtyped
    @override
    def forward(
        self, hash_indices: Int64[torch.Tensor, "batch seq total_heads"]
    ) -> Float[torch.Tensor, "batch seq total_heads dim_per_head"]:
        """
        Retrieves embedding vectors for pre-computed hash indices.
        Args:
            hash_indices: [batch_size, seq_len, total_heads]
        Returns:
            torch.Tensor: [batch_size, seq_len, total_heads, embedding_dim_per_head]
        """
        assert isinstance(self.offsets, torch.Tensor)
        shifted_indices = hash_indices.to(self.offsets.device) + self.offsets
        return self.embedding(shifted_indices)


@final
class EngramLayer(nn.Module):
    """
    Complete Engram Layer as described in Section 2.1-2.3 of the Engram paper.

    1. Extracts suffix N-grams (via CompressedTokenizer and MultiHeadHash)
    2. Computes indices via Multi-Head Hashing
    3. Retrieves vectors from K independent embedding tables
    4. Applies Context-Aware Gating modulation
    5. Residual connection to Transformer Block hidden states
    """

    config: EngramConfig
    layer_id: int
    compressor: CompressedTokenizer | None
    ngram_sizes: list[int]
    hash_heads: int
    num_branches: int
    kernel_size: int
    dilation: int
    hidden_dim: int
    total_embedding_dim: int
    embedding_dim_per_head: int
    hash_mapping: FixedNgramHashMapping | NgramHashMapping
    multi_head_embedding: MultiHeadEmbedding
    gating: ContextAwareGating | HeadFactorizedGating
    short_conv: ShortConv
    rq_multi_head_embedding: MultiHeadEmbedding | None
    arith_multi_head_embedding: MultiHeadEmbedding | None
    rq_gating: ContextAwareGating | None
    arith_gating: ContextAwareGating | None
    rq_short_conv: ShortConv | None
    arith_short_conv: ShortConv | None
    fusion_gate: nn.Sequential | None
    last_norm_ratio: float

    def __init__(
        self,
        config: EngramConfig,
        layer_id: int,
        primes: list[int],
        compressor: CompressedTokenizer | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the EngramLayer.

        Args:
            config: EngramConfig containing hyperparameters.
            layer_id: The ID of this layer.
            primes: List of pre-calculated primes for this layer's heads.
            compressor: Optional CompressedTokenizer for token mapping.
        """
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.compressor = compressor

        self.ngram_sizes = config.ngram_sizes
        self.hash_heads = config.n_head_per_ngram
        self.num_branches = config.hc_mult
        self.kernel_size = config.conv_kernel_size
        self.dilation = (
            config.conv_dilation
            if config.conv_dilation is not None
            else config.max_ngram_size
        )
        assert config.hidden_size is not None
        assert config.embedding_dim is not None
        self.hidden_dim = config.hidden_size

        self.total_embedding_dim = config.embedding_dim
        self.embedding_dim_per_head = self.total_embedding_dim // (
            len(self.ngram_sizes) * self.hash_heads
        )

        # 0. Hash Mapping
        # Map pad_id to compressed space for hashing consistency
        mapped_pad_id = config.pad_id
        if self.compressor is not None:
            assert config.pad_id is not None
            mapped_pad_id = self.compressor.map_id(config.pad_id)

        assert config.compressed_vocab_size is not None
        assert mapped_pad_id is not None

        mapping_type = (
            FixedNgramHashMapping
            if config.hash_backend == "arithmetic_fixed"
            else NgramHashMapping
        )
        self.hash_mapping = mapping_type(
            compressed_vocab_size=config.compressed_vocab_size,
            engram_vocab_size_per_ngram=config.engram_vocab_size_per_ngram,
            ngram_sizes=config.ngram_sizes,
            n_head_per_ngram=config.n_head_per_ngram,
            layer_ids=[layer_id],
            pad_id=mapped_pad_id,
            seed=config.seed,
        )

        self.rq_multi_head_embedding = None
        self.arith_multi_head_embedding = None
        self.rq_gating = None
        self.arith_gating = None
        self.rq_short_conv = None
        self.arith_short_conv = None
        self.fusion_gate = None
        self.register_buffer("rq_bucket_specificity", None, persistent=False)

        if config.hash_backend == "mixed_v2":
            n_sizes = len(self.ngram_sizes)
            r = config.n_rq_levels_used
            a = config.n_arith_heads_per_ngram
            rq_primes: list[int] = []
            arith_primes: list[int] = []
            pos = 0
            for _ in range(n_sizes):
                rq_primes += primes[pos : pos + r]
                pos += r
                arith_primes += primes[pos : pos + a]
                pos += a
            rq_embedding_dim_per_head = self.total_embedding_dim // (n_sizes * r)
            arith_embedding_dim_per_head = self.total_embedding_dim // (n_sizes * a)

            self.rq_multi_head_embedding = MultiHeadEmbedding(
                primes=rq_primes,
                embedding_dim_per_head=rq_embedding_dim_per_head,
                sparse=config.use_sparse_embeddings,
            )
            self.arith_multi_head_embedding = MultiHeadEmbedding(
                primes=arith_primes,
                embedding_dim_per_head=arith_embedding_dim_per_head,
                sparse=config.use_sparse_embeddings,
            )
            self.rq_gating = ContextAwareGating(
                config=config,
                engram_hidden_size=self.total_embedding_dim,
                hidden_size=self.hidden_dim,
                hc_mult=self.num_branches,
                zero_init=config.gating_zero_init,
            )
            self.arith_gating = ContextAwareGating(
                config=config,
                engram_hidden_size=self.total_embedding_dim,
                hidden_size=self.hidden_dim,
                hc_mult=self.num_branches,
                zero_init=config.gating_zero_init,
            )
            self.rq_short_conv = ShortConv(
                hidden_size=self.hidden_dim,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                hc_mult=self.num_branches,
                activation=True,
                zero_init=config.conv_zero_init,
            )
            self.arith_short_conv = ShortConv(
                hidden_size=self.hidden_dim,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                hc_mult=self.num_branches,
                activation=True,
                zero_init=config.conv_zero_init,
            )
            self.fusion_gate = nn.Sequential(
                nn.RMSNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, 1),
            )
            fusion_linear = self.fusion_gate[1]
            assert isinstance(fusion_linear, nn.Linear)
            nn.init.zeros_(fusion_linear.weight)
            nn.init.zeros_(fusion_linear.bias)

            # Kept for compatibility with code that expects these attributes.
            self.multi_head_embedding = self.rq_multi_head_embedding
            self.gating = self.rq_gating
            self.short_conv = self.rq_short_conv
        else:
            # 1. MultiHeadEmbedding
            self.multi_head_embedding = MultiHeadEmbedding(
                primes=primes,
                embedding_dim_per_head=self.embedding_dim_per_head,
                sparse=config.use_sparse_embeddings,
            )

            # 2. Context-Aware Gating
            if config.memory_fusion == "flatten":
                self.gating = ContextAwareGating(
                    config=config,
                    engram_hidden_size=self.total_embedding_dim,
                    hidden_size=self.hidden_dim,
                    hc_mult=self.num_branches,
                    zero_init=config.gating_zero_init,
                )
            elif config.memory_fusion == "head_factorized":
                self.gating = HeadFactorizedGating(
                    config=config,
                    num_heads=len(primes),
                    embedding_dim_per_head=self.embedding_dim_per_head,
                    hidden_size=self.hidden_dim,
                    hc_mult=self.num_branches,
                    zero_init=config.gating_zero_init,
                )
                if config.head_router_selection in {"specificity", "rq_snr", "rq_signal"}:
                    if config.hash_backend != "rq":
                        raise ValueError(
                            "address-score head selection currently requires hash_backend='rq'"
                        )
                    if not config.rq_table_dir:
                        raise ValueError(
                            "address-score head selection requires rq_table_dir"
                        )
                    rq_mapping = RQNgramMapping(config.rq_table_dir)
                    if config.head_router_selection == "rq_snr":
                        score_array = rq_mapping.signal_to_interference_table()
                    elif config.head_router_selection == "rq_signal":
                        score_array = rq_mapping.residual_signal_table()
                    else:
                        rows: list[np.ndarray] = []
                        for n in rq_mapping.ngram_sizes:
                            codes = rq_mapping.codes[n]
                            for level in range(rq_mapping.num_levels):
                                counts = np.bincount(
                                    codes[:, level], minlength=rq_mapping.codebook_size
                                ).astype(np.float32)
                                # Distinct n-grams per bucket measure collision load. The
                                # per-head z-score makes levels comparable without using
                                # any downstream labels or evaluation examples.
                                score = -np.log1p(counts)
                                score = (score - score.mean()) / (score.std() + 1e-6)
                                rows.append(score)
                        score_array = np.stack(rows)
                    specificity = torch.from_numpy(score_array).float()
                    if specificity.shape != (len(primes), max(primes)):
                        raise ValueError(
                            "RQ specificity table shape does not match memory heads: "
                            f"{tuple(specificity.shape)} vs {(len(primes), max(primes))}"
                        )
                    self.rq_bucket_specificity = specificity
            else:
                raise ValueError(
                    "memory_fusion must be 'flatten' or 'head_factorized', "
                    f"got {config.memory_fusion!r}"
                )

            # 3. ShortConv
            self.short_conv = ShortConv(
                hidden_size=self.hidden_dim,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                hc_mult=self.num_branches,
                activation=True,
                zero_init=config.conv_zero_init,
            )
        self.last_norm_ratio = 0.0  # Default to zero

    @property
    def value_proj(self) -> nn.Linear:
        return self.gating.w_v

    @property
    def key_projs(self) -> nn.ModuleList:
        return self.gating.w_k

    @property
    def norm1(self) -> nn.ModuleList:
        return self.gating.norm_k

    @property
    def norm2(self) -> nn.ModuleList:
        return self.gating.norm_h

    @jaxtyped
    @override
    def forward(
        self,
        input_ids: Int64[torch.Tensor, "batch seq"] | None = None,
        compressed_ids: Int64[torch.Tensor, "batch seq"] | None = None,
        hidden_states: Float[torch.Tensor, "batch seq hidden_dim"]
        | Float[torch.Tensor, "batch seq hc_mult hidden_dim"]
        | None = None,
        engram_hash_indices: Any = None,
    ) -> (
        Float[torch.Tensor, "batch seq hidden_dim"]
        | Float[torch.Tensor, "batch seq hc_mult hidden_dim"]
    ):
        """
        Forward pass of the EngramLayer.

        Args:
            input_ids: [batch_size, seq_len] Original token IDs.
            compressed_ids: [batch_size, seq_len] Compressed token IDs.
            hidden_states: [batch_size, seq_len, hidden_dim] or [B, L, M, D].
            engram_hash_indices: Optional precomputed hash indices [B, L, total_heads].

        Returns:
            torch.Tensor: Modified hidden states with Engram contributions.
        """
        if hidden_states is None:
            raise ValueError("hidden_states must be provided to EngramLayer.forward()")

        if engram_hash_indices is None:
            if input_ids is None:
                raise ValueError(
                    "Either engram_hash_indices or input_ids must be provided."
                )
            if self.compressor is None:
                raise ValueError(
                    "Compressor must be provided to compute hashes from input_ids."
                )
            # Step 1: Compress and hash
            c_ids = self.compressor.compress(input_ids)
            hashes_np = self.hash_mapping.hash(c_ids)[self.layer_id]
            engram_hash_indices = safe_from_numpy(hashes_np).to(hidden_states.device)

        # Ensure hidden_states is [B, L, M, D] if it's not already
        is_3d = hidden_states.dim() == 3
        if is_3d:
            hidden_states_m = hidden_states.unsqueeze(2).expand(
                -1, -1, self.num_branches, -1
            )
        else:
            hidden_states_m = hidden_states

        if self.config.hash_backend == "mixed_v2":
            if not isinstance(engram_hash_indices, dict):
                raise ValueError("mixed_v2 expects {'rq': ..., 'arith': ...} hash indices")
            assert self.rq_multi_head_embedding is not None
            assert self.arith_multi_head_embedding is not None
            assert self.rq_gating is not None
            assert self.arith_gating is not None
            assert self.rq_short_conv is not None
            assert self.arith_short_conv is not None
            assert self.fusion_gate is not None

            rq_embeddings = self.rq_multi_head_embedding(
                engram_hash_indices["rq"]
            ).flatten(start_dim=-2)
            arith_embeddings = self.arith_multi_head_embedding(
                engram_hash_indices["arith"]
            ).flatten(start_dim=-2)

            y_rq = self.rq_short_conv(self.rq_gating(rq_embeddings, hidden_states_m))
            y_arith = self.arith_short_conv(
                self.arith_gating(arith_embeddings, hidden_states_m)
            )

            gate_source = hidden_states if is_3d else hidden_states_m.mean(dim=2)
            fusion = torch.sigmoid(self.fusion_gate(gate_source)).unsqueeze(2)
            y = fusion * y_rq + (1.0 - fusion) * y_arith

            if is_3d:
                if self.num_branches == 1:
                    y = y.squeeze(2)
                else:
                    y = y.sum(dim=2)

            if self.config.enable_telemetry:
                with torch.no_grad():
                    y_norm = cast(
                        "torch.Tensor", torch.linalg.vector_norm(y.float(), ord=2)
                    )
                    h_norm = cast(
                        "torch.Tensor",
                        torch.linalg.vector_norm(hidden_states.float(), ord=2),
                    )
                    self.last_norm_ratio = (y_norm / (h_norm + 1e-6)).item()

            return (hidden_states + y).to(hidden_states.dtype)

        # Step 4: Context-Aware Gating modulation
        # Step 1: Retrieve vectors.  The reference fusion flattens heads; CREDIT keeps
        # the head axis to expose independently routable, parameter-matched blocks.
        all_embeddings = self.multi_head_embedding(engram_hash_indices)
        selection_scores = None
        if isinstance(self.gating, HeadFactorizedGating):
            e_t = all_embeddings.to(hidden_states.device)
            if self.rq_bucket_specificity is not None:
                score_table = self.rq_bucket_specificity.to(
                    device=engram_hash_indices.device
                )
                head_ids = torch.arange(
                    engram_hash_indices.shape[-1], device=engram_hash_indices.device
                ).view(1, 1, -1)
                selection_scores = score_table[head_ids, engram_hash_indices]
        else:
            e_t = all_embeddings.flatten(start_dim=-2).to(hidden_states.device)

        # Step 4: Context-Aware Gating modulation
        # gated_value has shape [B, L, M, D]
        if isinstance(self.gating, HeadFactorizedGating):
            gated_value = self.gating(
                e_t, hidden_states_m, head_selection_scores=selection_scores
            )
        else:
            gated_value = self.gating(e_t, hidden_states_m)

        # Step 5: ShortConv module
        # y has shape [B, L, M, D]
        y = self.short_conv(gated_value)

        # Step 6: Residual connection to Transformer Block hidden states
        if is_3d:
            if self.num_branches == 1:
                y = y.squeeze(2)
            else:
                # Sum branches if no out_proj is provided
                y = y.sum(dim=2)

        if self.config.enable_telemetry:
            with torch.no_grad():
                y_norm = cast(
                    "torch.Tensor", torch.linalg.vector_norm(y.float(), ord=2)
                )
                h_norm = cast(
                    "torch.Tensor",
                    torch.linalg.vector_norm(hidden_states.float(), ord=2)
                    if is_3d
                    else torch.linalg.vector_norm(hidden_states_m.float(), ord=2),
                )
                self.last_norm_ratio = (y_norm / (h_norm + 1e-8)).item()

        # Final result matches the input hidden_states shape
        return (hidden_states + y).to(hidden_states.dtype)
