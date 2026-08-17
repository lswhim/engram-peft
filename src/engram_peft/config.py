# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast, final, override

from transformers import PretrainedConfig as PreTrainedConfig


@dataclass(kw_only=True)
@final
class EngramConfig(PreTrainedConfig):
    """Configuration class for Engram PEFT module.

    This class inherits from transformers.PretrainedConfig and uses python
    dataclasses for defining Engram-specific hyperparameters exactly matching
    the specifications in the Engram paper Appendix A Table 5.
    """

    model_type: ClassVar[str] = "engram"

    engram_vocab_size_per_ngram: list[int] = field(
        default_factory=lambda: [1131200, 1131200],
        metadata={
            "help": "Total number of hash buckets for each N-gram size (e.g., [2, 3]). "
            + "Matches Engram paper Appendix A Table 5."
        },
    )
    ngram_sizes: list[int] = field(
        default_factory=lambda: [2, 3],
        metadata={"help": "Explicit list of N-gram sizes to use. Default is [2, 3]."},
    )
    max_ngram_size: int = field(
        default=3,
        metadata={"help": "Maximum N-gram size. Automatically derived if not set."},
    )
    n_head_per_ngram: int = field(
        default=8,
        metadata={"help": "Number of hashing heads per N-gram size."},
    )
    embedding_dim: int = field(
        default=1280,
        metadata={"help": "Internal dimension of Engram embeddings before projection."},
    )
    enable_tokenizer_compression: bool = field(
        default=True,
        metadata={
            "help": "Whether to use surjective mapping V -> V' to reduce token space."
        },
    )
    layer_container_path: str | None = field(
        default=None,
        metadata={
            "help": "[Auto-Detect] Dot-separated path to the transformer layers (e.g. 'model.layers')."
        },
    )
    target_layers: list[int] = field(
        default_factory=lambda: [1, 14],
        metadata={
            "help": "List of layer indices where Engram layers will be injected. 0-indexed. Which is [2, 15] in paper."
        },
    )
    target_modules: list[str] | str | None = field(
        default=None,
        metadata={"help": "Reserved for future use (similar to LoRA target_modules)."},
    )
    hc_mult: int = field(
        default=4,
        metadata={
            "help": "Hyper-constant multiplier for hashing logic. See Engram paper."
        },
    )
    combine_mhc: bool = field(
        default=True,
        metadata={
            "help": "Whether to combine Multi-Head features via summation or concatenation."
        },
    )
    hidden_size: int | None = field(
        default=None,
        metadata={
            "help": "[Auto-Detect] The hidden dimension of the base model (e.g. 2048)."
        },
    )
    conv_kernel_size: int = field(
        default=4,
        metadata={"help": "Kernel size for the Engram convolution block."},
    )
    conv_dilation: int | None = field(
        default=None,
        metadata={
            "help": "Dilation for the convolution block. Defaults to max_ngram_size."
        },
    )
    conv_zero_init: bool = field(
        default=True,
        metadata={
            "help": "Whether to initialize convolution weights with zeros for identity behavior."
        },
    )
    gating_zero_init: bool = field(
        default=True,
        metadata={"help": "Whether to initialize gating parameters with zeros."},
    )
    learning_rate_multiplier: float = field(
        default=5.0,
        metadata={
            "help": "LR multiplier for Engram base weights relative to the base LR."
        },
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay for Engram parameters."},
    )
    tokenizer_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": "[Recommended] Tokenizer identifier to build compression mapping."
        },
    )
    compressed_vocab_size: int | None = field(
        default=None,
        metadata={
            "help": "[Persistence] The size of the resolved hashing vocabulary. Stored after first init."
        },
    )
    pad_id: int | None = field(
        default=None,
        metadata={"help": "[Auto-Detect] The padding token ID of the tokenizer/model."},
    )
    seed: int = field(
        default=0,
        metadata={"help": "Random seed for deterministic hashing primes."},
    )
    hash_backend: str = field(
        default="arithmetic",
        metadata={
            "help": "Address backend: 'arithmetic' (original XOR-multiply hash), "
            + "'arithmetic_fixed' (exact equal-K arithmetic control), "
            + "'rq' (frozen offline RQ semantic-hash table, see scripts/build_rq_table.py), "
            + "'mixed' (per-head split: n_rq_levels_used RQ + n_arith_heads_per_ngram arith), "
            + "or 'mixed_v2' (separate RQ/arith branches fused by a token-level gate)."
        },
    )
    rq_table_dir: str | None = field(
        default=None,
        metadata={"help": "Path to the prebuilt RQ table dir (required if hash_backend in {'rq','mixed'})."},
    )
    rq_cache_dir: str | None = field(
        default=None,
        metadata={
            "help": "Writable persistent cache for lazy semantic RQ codes. Required "
            "for strict all-n-gram rq runs; use a unique directory per address geometry."
        },
    )
    rq_embed_device: str = field(
        default="cuda",
        metadata={"help": "Device used for first-seen n-gram semantic encoding."},
    )
    rq_embed_batch_size: int = field(
        default=256,
        metadata={"help": "Unique cache misses per frozen semantic-encoder batch."},
    )
    n_rq_levels_used: int = field(
        default=4,
        metadata={"help": "Mixed-hash: how many of the RQ table's M levels to use per n-gram size."},
    )
    n_arith_heads_per_ngram: int = field(
        default=4,
        metadata={"help": "Mixed-hash: how many arith heads per n-gram size (in addition to n_rq_levels_used)."},
    )
    clip_grad_per_group: bool = field(
        default=False,
        metadata={
            "help": "Whether to use group-wise gradient clipping (Backbone vs Engram) instead of global norm clipping."
        },
    )
    enable_telemetry: bool = field(
        default=False,
        metadata={
            "help": "Whether to collect deep metrics (norm, max, zero-rate) for all parameter groups during training."
        },
    )
    entropy_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for the gating entropy penalty loss."},
    )
    memory_fusion: str = field(
        default="flatten",
        metadata={
            "help": "Memory fusion: 'flatten' reproduces Engram; "
            "'head_factorized' assigns independent utility gates to matched-parameter head blocks."
        },
    )
    head_router_top_k: int = field(
        default=0,
        metadata={
            "help": "For head_factorized fusion, keep the top-k heads per token/branch. "
            "0 keeps all heads with soft gates."
        },
    )
    head_router_preserve_mass: bool = field(
        default=False,
        metadata={
            "help": "Rescale sparse/forced routes to preserve the all-head gate mass, "
            "isolating head selection from total memory strength."
        },
    )
    head_router_selection: str = field(
        default="context",
        metadata={
            "help": "Score used to select sparse memory heads: 'context' uses the "
            "learned Engram route logits; 'specificity' uses the inverse collision "
            "load of the addressed RQ bucket. Gate amplitudes remain context-derived."
        },
    )
    head_router_use_null: bool = field(
        default=False,
        metadata={
            "help": "Allow an explicit no-memory route when every head logit is below "
            "head_router_null_threshold."
        },
    )
    head_router_null_threshold: float = field(
        default=0.0,
        metadata={"help": "Fixed logit threshold for the no-memory route."},
    )
    credit_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Weight of counterfactual equal-budget route preference loss."},
    )
    credit_pair_fraction: float = field(
        default=0.5,
        metadata={"help": "Fraction of each training batch used for route-pair credit."},
    )
    credit_route_k: int = field(
        default=4,
        metadata={"help": "Number of active memory heads in each counterfactual route."},
    )
    credit_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for route-utility preference calibration."},
    )
    backbone_freeze_steps: int = field(
        default=0,
        metadata={
            "help": "Number of initial steps to freeze the backbone for 'Adapter-First' pre-training."
        },
    )
    engram_dtype: str | None = field(
        default=None,
        metadata={
            "help": "Explicit data type for Engram parameters (e.g., 'float32', 'float16', 'bfloat16'). "
            + "If None, it will be automatically detected from the backbone's compute_dtype."
        },
    )
    use_sparse_embeddings: bool = field(
        default=True,
        metadata={
            "help": "Whether to use sparse embeddings for Engram layers. Enabled by default for performance. "
            + "Note: Standard HF Trainers (SFTTrainer) do not support sparse gradients and require this to be False."
        },
    )
    engram_version: str = field(
        default="1.2.4",
        metadata={"help": "The version of the Engram state file format."},
    )
    train_mode: str | None = field(
        default=None,
        metadata={
            "help": "Training mode: 'engram_only', 'preserve_trainable', or 'full_finetune'."
        },
    )
    wrap_peft: bool = field(
        default=False,
        metadata={"help": "Whether the base model is a PEFT model."},
    )

    def __init__(
        self,
        engram_vocab_size_per_ngram: list[int] | None = None,
        ngram_sizes: list[int] | None = None,
        n_head_per_ngram: int = 8,
        embedding_dim: int = 1280,
        enable_tokenizer_compression: bool = True,
        target_layers: list[int] | None = None,
        hc_mult: int = 4,
        combine_mhc: bool = True,
        conv_kernel_size: int = 4,
        conv_dilation: int | None = None,
        conv_zero_init: bool = True,
        gating_zero_init: bool = True,
        learning_rate_multiplier: float = 5.0,
        weight_decay: float = 0.0,
        tokenizer_name_or_path: str | None = None,
        compressed_vocab_size: int | None = None,
        pad_id: int | None = None,
        seed: int = 0,
        hidden_size: int | None = None,
        clip_grad_per_group: bool = False,
        enable_telemetry: bool = False,
        entropy_loss_weight: float = 0.0,
        memory_fusion: str = "flatten",
        head_router_top_k: int = 0,
        head_router_preserve_mass: bool = False,
        head_router_selection: str = "context",
        head_router_use_null: bool = False,
        head_router_null_threshold: float = 0.0,
        credit_loss_weight: float = 0.0,
        credit_pair_fraction: float = 0.5,
        credit_route_k: int = 4,
        credit_temperature: float = 1.0,
        backbone_freeze_steps: int = 0,
        engram_dtype: str | None = None,
        use_sparse_embeddings: bool = True,
        engram_version: str = "1.2.4",
        train_mode: str | None = None,
        wrap_peft: bool = False,
        **kwargs: Any,
    ):
        """Constructs EngramConfig."""
        self.engram_vocab_size_per_ngram = (
            engram_vocab_size_per_ngram
            if engram_vocab_size_per_ngram is not None
            else [1131200, 1131200]
        )
        self.ngram_sizes = ngram_sizes if ngram_sizes is not None else [2, 3]
        self.max_ngram_size = max(self.ngram_sizes)
        self.n_head_per_ngram = n_head_per_ngram
        self.embedding_dim = embedding_dim
        self.enable_tokenizer_compression = enable_tokenizer_compression
        self.target_layers = target_layers if target_layers is not None else [1, 14]
        self.hc_mult = hc_mult
        self.combine_mhc = combine_mhc
        self.conv_kernel_size = conv_kernel_size

        # dynamic default for conv_dilation
        self.conv_dilation = (
            conv_dilation if conv_dilation is not None else self.max_ngram_size
        )

        self.conv_zero_init = conv_zero_init
        self.gating_zero_init = gating_zero_init
        self.learning_rate_multiplier = learning_rate_multiplier
        self.hidden_size = hidden_size
        self.weight_decay = weight_decay
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.compressed_vocab_size = compressed_vocab_size
        self.pad_id = pad_id
        self.seed = seed
        self.clip_grad_per_group = clip_grad_per_group
        self.enable_telemetry = enable_telemetry
        self.entropy_loss_weight = entropy_loss_weight
        self.memory_fusion = memory_fusion
        self.head_router_top_k = head_router_top_k
        self.head_router_preserve_mass = head_router_preserve_mass
        self.head_router_selection = head_router_selection
        self.head_router_use_null = head_router_use_null
        self.head_router_null_threshold = head_router_null_threshold
        self.credit_loss_weight = credit_loss_weight
        self.credit_pair_fraction = credit_pair_fraction
        self.credit_route_k = credit_route_k
        self.credit_temperature = credit_temperature
        self.backbone_freeze_steps = backbone_freeze_steps
        self.engram_dtype = engram_dtype
        self.use_sparse_embeddings = use_sparse_embeddings
        self.engram_version = engram_version
        self.train_mode = train_mode
        self.wrap_peft = wrap_peft

        super().__init__(**kwargs)
        # Ensure extra kwargs are set, as PretrainedConfig might miss them in some environments
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)

    @override
    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs: Any) -> EngramConfig:
        """Instantiates an EngramConfig from a python dictionary."""
        return cast("EngramConfig", super().from_dict(config_dict, **kwargs))

    @override
    def to_dict(self) -> dict[str, Any]:
        """Serializes this instance to a Python dictionary."""
        output = super().to_dict()

        # Override output with dataclass explicit values to capture our specific attributes
        for field_def in dataclasses.fields(self):
            output[field_def.name] = getattr(self, field_def.name)

        output["model_type"] = self.model_type
        return output

    @override
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[Any],
        cache_dir: str | os.PathLike[Any] | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        **kwargs: Any,
    ) -> EngramConfig:
        """Loads from JSON file/pretrained repo."""
        return cast(
            "EngramConfig",
            super().from_pretrained(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                **kwargs,
            ),
        )

    @override
    def save_pretrained(
        self,
        save_directory: str | os.PathLike[Any],
        push_to_hub: bool = False,
        **kwargs: Any,
    ) -> None:
        """Saves as JSON file."""
        os.makedirs(save_directory, exist_ok=True)
        super().save_pretrained(save_directory, push_to_hub=push_to_hub, **kwargs)
