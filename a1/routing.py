"""How evenly does the router spread tokens over the experts?

    python a1/routing.py           # writes results/a1/routing.csv and routing.png

Counts the tokens each expert receives in each layer, for the pretrained
weights on real text and for random weights.
"""
from common import ROOT, require_gpu, save_plot, write_csv

import torch
from tokenizers import Tokenizer

from olmoe import OLMOE_1B_7B, build_model, load_model, make_seqs
from olmoe.load import default_weights_path

TEXT = ROOT / "assignments" / "a1.md"  # any English text will do


@torch.no_grad()
def expert_counts(model, ids):
    """How many tokens each layer's router sends to each expert: [n_layers, n_experts]."""
    counts = []
    E, k = model.cfg.n_experts, model.cfg.top_k

    def hook(router, args, logits):  # logits: [T, n_experts]
        chosen = logits.topk(k, dim=-1).indices
        counts.append(torch.bincount(chosen.flatten(), minlength=E).cpu())

    handles = [layer.moe.router.register_forward_hook(hook) for layer in model.layers]
    model(ids, make_seqs([len(ids)]))
    for h in handles:
        h.remove()
    return torch.stack(counts).float()  # [n_layers, n_experts]


def main():
    require_gpu()
    tok = Tokenizer.from_file(str(default_weights_path() / "tokenizer.json"))
    ids = torch.tensor(tok.encode(TEXT.read_text()).ids[:4096], device="cuda")  # [T]: one sequence
    print(f"{len(ids)} tokens from {TEXT.name}")
    counts = {"pretrained": expert_counts(load_model().eval(), ids)}
    torch.cuda.empty_cache()
    counts["random"] = expert_counts(build_model(OLMOE_1B_7B).eval(), ids)

    rows = []
    for name, c in counts.items():
        for layer, row in enumerate(c):
            rows.append(dict(weights=name, layer=layer, max_over_mean=round((row.max() / row.mean()).item(), 3),
                             min_over_mean=round((row.min() / row.mean()).item(), 3),
                             unused_experts=int((row == 0).sum())))
        print(f"{name:10s} busiest expert gets {c.max(1).values.div(c.mean(1)).mean():.2f}x the mean load "
              f"(averaged over layers)")
    write_csv("routing.csv", rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    for ax, (name, c) in zip(axes, counts.items()):
        im = ax.imshow(c / c.mean(1, keepdim=True), aspect="auto", cmap="viridis", vmin=0, vmax=3)
        ax.set(title=f"{name} weights", xlabel="expert", ylabel="layer")
    fig.colorbar(im, ax=axes, label="tokens routed to the expert / mean")
    save_plot(fig, "routing.png")


if __name__ == "__main__":
    main()
