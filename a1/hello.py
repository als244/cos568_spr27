"""Experiment 1: load OLMoE and generate a few tokens greedily.

    python a1/hello.py "The capital of France is"

There is no KV cache here, so every new token recomputes the whole sequence.
Real inference engines (Assignment 3) avoid that.
"""
import sys
import time

from common import require_gpu

import torch
from tokenizers import Tokenizer

from olmoe import load_model, make_seqs
from olmoe.load import default_weights_path


@torch.no_grad()
def main():
    require_gpu()
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
    tok = Tokenizer.from_file(str(default_weights_path() / "tokenizer.json"))
    t0 = time.time()
    model = load_model().eval()
    print(f"loaded {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters "
          f"in {time.time() - t0:.0f} s; {torch.cuda.memory_allocated() / 2**30:.1f} GiB on the GPU")
    ids = torch.tensor(tok.encode(prompt).ids, device="cuda")  # [T]: one sequence
    for _ in range(20):
        logits = model(ids, make_seqs([len(ids)]))               # [T, vocab]
        ids = torch.cat([ids, logits[-1].argmax(keepdim=True)])
    print(tok.decode(ids.tolist()))


if __name__ == "__main__":
    main()
