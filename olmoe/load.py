"""Load and save checkpoints in HuggingFace format.

save_hf() writes the same layout the official checkpoint uses, so a model
you fine-tune in Assignment 2 loads directly in vLLM and SGLang in Assignment 3.
"""
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .config import OlmoeConfig
from .model import Olmoe, build_model

_TOP = {
    "model.embed_tokens.weight": "embed.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "lm_head.weight",
}
_LAYER = {
    "input_layernorm.weight": "attn_norm.weight",
    "self_attn.q_proj.weight": "attn.q_proj.weight",
    "self_attn.k_proj.weight": "attn.k_proj.weight",
    "self_attn.v_proj.weight": "attn.v_proj.weight",
    "self_attn.o_proj.weight": "attn.o_proj.weight",
    "self_attn.q_norm.weight": "attn.q_norm.weight",
    "self_attn.k_norm.weight": "attn.k_norm.weight",
    "post_attention_layernorm.weight": "moe_norm.weight",
    "mlp.gate.weight": "moe.router.weight",
}
_EXPERT = {"gate_proj": "w_gate", "up_proj": "w_up", "down_proj": "w_down"}
_EXTRA_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                "generation_config.json"]


def default_weights_path() -> Path:
    """The checkpoint directory that setup/env.sh points to ($OLMOE_WEIGHTS)."""
    p = os.environ.get("OLMOE_WEIGHTS")
    if not p:
        raise RuntimeError("OLMOE_WEIGHTS is not set; did you `source setup/env.sh`?")
    return Path(p)


def hf_to_ours(name: str, n_layers: int):
    """HF tensor name -> (our param name, expert index or None); None if layer is dropped."""
    if name in _TOP:
        return _TOP[name], None
    parts = name.split(".")  # model.layers.{i}.<rest>
    assert parts[:2] == ["model", "layers"], f"unexpected tensor {name}"
    i = int(parts[2])
    if i >= n_layers:
        return None
    rest = ".".join(parts[3:])
    if rest in _LAYER:
        return f"layers.{i}.{_LAYER[rest]}", None
    # model.layers.{i}.mlp.experts.{e}.{gate,up,down}_proj.weight
    assert parts[3:5] == ["mlp", "experts"], f"unexpected tensor {name}"
    return f"layers.{i}.moe.{_EXPERT[parts[6]]}", int(parts[5])


def load_model(path=None, n_layers=None, device="cuda", dtype=torch.bfloat16, **kw) -> Olmoe:
    """Load a HF-format OLMoE checkpoint, optionally keeping only the first n_layers.
    Extra keyword arguments (recompute, moe_impl) go to Olmoe."""
    path = Path(path) if path else default_weights_path()
    cfg = OlmoeConfig.from_hf(path)
    if n_layers is not None:
        cfg = cfg.with_(n_layers=n_layers)
    model = build_model(cfg, device, dtype, init=False, **kw)
    params = dict(model.named_parameters())
    expected = {(n, e) for n, p in params.items()
                for e in (range(p.shape[0]) if ".moe.w_" in n else [None])}
    loaded = set()
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(f, framework="pt") as st:
            for hf_name in st.keys():
                target = hf_to_ours(hf_name, cfg.n_layers)
                if target is None:
                    continue
                name, e = target
                p = params[name] if e is None else params[name][e]
                with torch.no_grad():
                    p.copy_(st.get_tensor(hf_name))
                loaded.add(target)
    missing = expected - loaded
    assert not missing, f"{len(missing)} tensors missing from checkpoint, e.g. {next(iter(missing))}"
    return model


def hf_state_items(model: Olmoe):
    """Yield (HF name, tensor) pairs for every weight in the model."""
    inv_top = {v: k for k, v in _TOP.items()}
    inv_layer = {v: k for k, v in _LAYER.items()}
    inv_expert = {v: k for k, v in _EXPERT.items()}
    for name, p in model.named_parameters():
        if name in inv_top:
            yield inv_top[name], p
            continue
        _, i, rest = name.split(".", 2)  # layers.{i}.<rest>
        if rest in inv_layer:
            yield f"model.layers.{i}.{inv_layer[rest]}", p
        else:  # moe.w_{gate,up,down}: one HF tensor per expert
            proj = inv_expert[rest.split(".")[1]]
            for e in range(p.shape[0]):
                yield f"model.layers.{i}.mlp.experts.{e}.{proj}.weight", p[e]


@torch.no_grad()
def save_hf(model: Olmoe, out_dir, src_dir=None, shard_bytes=5 * 2**30):
    """Save in HF format: sharded safetensors + index + config + tokenizer files."""
    out_dir, src_dir = Path(out_dir), Path(src_dir) if src_dir else default_weights_path()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Plan the shards first, then copy and write one shard at a time, so at
    # most shard_bytes of weights sit in CPU memory.
    shards, size = [[]], 0
    for name, t in hf_state_items(model):
        nbytes = t.numel() * t.element_size()
        if shards[-1] and size + nbytes > shard_bytes:
            shards.append([])
            size = 0
        shards[-1].append((name, t))
        size += nbytes
    weight_map, total = {}, 0
    for i, shard in enumerate(shards):
        fname = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file({name: t.detach().to("cpu", copy=True) for name, t in shard},
                  out_dir / fname, metadata={"format": "pt"})
        for name, t in shard:
            weight_map[name] = fname
            total += t.numel() * t.element_size()
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    hf_cfg = json.loads((src_dir / "config.json").read_text())
    hf_cfg["num_hidden_layers"] = model.cfg.n_layers
    hf_cfg["torch_dtype"] = str(next(model.parameters()).dtype).removeprefix("torch.")
    (out_dir / "config.json").write_text(json.dumps(hf_cfg, indent=2))
    for f in _EXTRA_FILES:
        if (src_dir / f).exists():
            shutil.copy(src_dir / f, out_dir / f)
