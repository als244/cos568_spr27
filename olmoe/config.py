"""Model configuration. Defaults match allenai/OLMoE-1B-7B-0924."""
import json
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class OlmoeConfig:
    """The shape of an OLMoE model. The defaults are OLMoE-1B-7B."""

    vocab_size: int = 50304
    hidden_size: int = 2048  # D in shape comments
    intermediate_size: int = 1024  # per expert
    n_layers: int = 16
    n_heads: int = 16
    n_experts: int = 64
    top_k: int = 8
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.n_heads

    def with_(self, **kw) -> "OlmoeConfig":
        """A copy with some fields changed, such as cfg.with_(n_layers=4)."""
        return replace(self, **kw)

    @classmethod
    def from_hf(cls, path) -> "OlmoeConfig":
        """Read the config.json of a HuggingFace checkpoint directory."""
        c = json.loads((Path(path) / "config.json").read_text())
        # Features of the HF Olmoe class that this checkpoint does not use.
        assert c.get("num_key_value_heads", c["num_attention_heads"]) == c["num_attention_heads"]
        assert not c.get("norm_topk_prob", False)
        assert c.get("clip_qkv") is None and not c.get("tie_word_embeddings", False)
        return cls(
            vocab_size=c["vocab_size"],
            hidden_size=c["hidden_size"],
            intermediate_size=c["intermediate_size"],
            n_layers=c["num_hidden_layers"],
            n_heads=c["num_attention_heads"],
            n_experts=c["num_experts"],
            top_k=c["num_experts_per_tok"],
            rope_theta=c.get("rope_theta", 10000.0),
            norm_eps=c.get("rms_norm_eps", 1e-5),
        )


OLMOE_1B_7B = OlmoeConfig()

# A tiny config with the same structure, for fast CPU tests.
TINY = OlmoeConfig(vocab_size=512, hidden_size=64, intermediate_size=32,
                   n_layers=2, n_heads=4, n_experts=8, top_k=2)
