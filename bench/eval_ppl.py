"""WikiText-2 test perplexity in MLX, for a base model, MLX quantization, or a
trained LittleBit checkpoint running on our kernels.

    .venv/bin/python bench/eval_ppl.py --base Qwen/Qwen3-0.6B                    # fp16
    .venv/bin/python bench/eval_ppl.py --base Qwen/Qwen3-0.6B --quant 4          # MLX 4-bit
    .venv/bin/python bench/eval_ppl.py --base Qwen/Qwen3-0.6B --ckpt checkpoints/<dir>

Protocol (the usual GPTQ/LittleBit one): the test split joined with "\\n\\n",
tokenised once, cut into non-overlapping `--seqlen` windows; perplexity =
exp(mean next-token NLL over all windows).  The text file is the raw
wikitext-2-raw-v1 test split (see --data).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--ckpt", default=None, help="trained LittleBit checkpoint dir")
    ap.add_argument("--quant", type=int, default=0, help="MLX affine quantization bits (base model only)")
    ap.add_argument("--data", default="checkpoints/data/wikitext2_test.txt")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--max-windows", type=int, default=0, help="0 = all")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    args = ap.parse_args()
    dtype = getattr(mx, args.dtype)

    if args.ckpt:
        from littlebit_kernel.load import load_littlebit_checkpoint
        model, tok = load_littlebit_checkpoint(args.ckpt, base_model=args.base, dtype=dtype, verbose=False)
        label = f"littlebit:{os.path.basename(args.ckpt.rstrip('/'))}"
    else:
        from mlx_lm import load
        model, tok = load(args.base)
        model.set_dtype(dtype)
        label = args.base
        if args.quant:
            nn.quantize(model, group_size=64, bits=args.quant,
                        class_predicate=lambda p, m: isinstance(m, nn.Linear) and "lm_head" not in p
                        and "embed" not in p)
            label += f" q{args.quant}"

    ids = tok.encode(open(args.data).read())
    n = len(ids) // args.seqlen
    if args.max_windows:
        n = min(n, args.max_windows)
    nll, count, t0 = 0.0, 0, time.time()
    for w in range(n):
        x = mx.array(ids[w * args.seqlen:(w + 1) * args.seqlen])[None]
        logits = model(x[:, :-1]).astype(mx.float32)
        loss = nn.losses.cross_entropy(logits, x[:, 1:], reduction="sum")
        nll += loss.item()
        count += args.seqlen - 1
    ppl = math.exp(nll / count)
    res = dict(model=label, ppl=ppl, windows=n, seqlen=args.seqlen, dtype=args.dtype, seconds=time.time() - t0)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
