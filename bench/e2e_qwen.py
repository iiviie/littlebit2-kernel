"""End-to-end decode speed of a Qwen3 model (default 0.6B, `--model`) in mlx-lm: fp16 vs 4/2-bit vs LittleBit.

    .venv/bin/python bench/e2e_qwen.py --variants fp16,q4,q2,lb0.1,lb0.55,lb1.0
    (wrap in bench/quiet.sh for clean numbers)

LittleBit variants replace every q/k/v/o/gate/up/down projection with
`LittleBitLinear` (embeddings + LM head stay fp16; Qwen3-0.6B ties them).
The factors come from the SVD (+ optional ITQ, `--itq`) initialisation
*without* QAT, so the generated text is gibberish -- but the layer shapes,
ranks and therefore the speed and memory are those of a real checkpoint.
`--init random` skips the SVD (faster setup, identical speed).

Reported per variant: decode tokens/s (mlx-lm's generation_tps over
`--tokens` generated tokens, median of `--reps` runs), prompt tokens/s,
bytes of linear-layer weights, and MLX peak memory during generation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from littlebit_kernel.format import LittleBitLayer, split_dim_for  # noqa: E402
from littlebit_kernel.init import littlebit_from_dense  # noqa: E402
from littlebit_kernel.nn import DEFAULT_TARGETS, group_shared_inputs, littlebitify  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import quiet_trial  # noqa: E402  (redo a rep if another GPU job overlapped)

MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "Write a short story about a lighthouse keeper who finds a message in a bottle."


def linear_bytes(model) -> int:
    """Bytes held by the replaced projections (whatever their format)."""
    total = 0
    for name, mod in model.named_modules():
        if name.split(".")[-1] in DEFAULT_TARGETS:
            total += sum(v.nbytes for _, v in nn.utils.tree_flatten(mod.parameters()))
    return total


def build(variant: str, args):
    from mlx_lm import load
    if variant.startswith(("ckpt:", "ckpt0:")):   # trained upstream-format checkpoint dir
        from littlebit_kernel.load import load_littlebit_checkpoint   # (ckpt0: = without grouping)
        dtype = mx.float16 if args.dtype == "float16" else mx.bfloat16
        kind, path = variant.split(":", 1)
        return load_littlebit_checkpoint(path, base_model=args.model, dtype=dtype, verbose=True,
                                         group=(kind == "ckpt"))
    model, tok = load(args.model)
    if args.dtype == "float16":
        model.set_dtype(mx.float16)   # (bf16 checkpoints: converting a 4B+ model
                                      #  briefly holds both copies; use --dtype bfloat16)
    if variant == "fp16":
        pass
    elif variant in ("q4", "q2", "q3", "q8"):
        bits = int(variant[1:])
        nn.quantize(model, group_size=64, bits=bits,
                    class_predicate=lambda p, m: isinstance(m, nn.Linear) and p.split(".")[-1] in DEFAULT_TARGETS)
    elif variant.startswith("lb"):
        bpw = float(variant[2:].rstrip("g"))   # "lb0.55g" = grouped q/k/v + gate/up
        res = args.residual

        def make(name, W, b):
            if args.init == "random":
                out_f, in_f = W.shape
                return LittleBitLayer.random(in_f, out_f, eff_bit=bpw, residual=res, seed=hash(name) % 2**31)
            return littlebit_from_dense(W, bpw, residual=res, use_itq=args.itq, itq_iters=10, bias=b)

        t0 = time.time()
        n = littlebitify(model, make, verbose=True)
        print(f"  {n} layers -> LittleBit @ {bpw} bpw ({args.init} init) in {time.time() - t0:.0f}s")
        if variant.endswith("g"):
            print(f"  grouped {group_shared_inputs(model)} q/k/v + gate/up groups")
    else:
        raise ValueError(variant)
    mx.eval(model.parameters())
    return model, tok


def run_variant(variant: str, args):
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    model, tok = build(variant, args)
    lin_bytes = linear_bytes(model)
    prompt = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True,
                                     enable_thinking=False) if hasattr(tok, "apply_chat_template") else PROMPT
    sampler = make_sampler(temp=0.0)
    gen, pre, text = [], [], ""
    # warmup (compiles kernels)
    for _ in stream_generate(model, tok, prompt, max_tokens=8, sampler=sampler):
        pass
    mx.reset_peak_memory()
    def one_rep():
        last, text = None, ""
        for last in stream_generate(model, tok, prompt, max_tokens=args.tokens, sampler=sampler):
            text += last.text
        return last, text

    for rep in range(args.reps):
        last, text = quiet_trial(one_rep)
        gen.append(last.generation_tps)
        pre.append(last.prompt_tps)
    rec = dict(variant=variant, decode_tps=statistics.median(gen), decode_tps_all=gen,
               prompt_tps=statistics.median(pre), generated_tokens=last.generation_tokens,
               linear_weight_mb=lin_bytes / 2**20, peak_mem_mb=mx.get_peak_memory() / 2**20,
               sample=text[:200])
    print(f"{variant:8s} decode {rec['decode_tps']:7.1f} tok/s  prompt {rec['prompt_tps']:7.0f} tok/s  "
          f"linear weights {rec['linear_weight_mb']:7.1f} MB  peak mem {rec['peak_mem_mb']:7.0f} MB")
    print(f"         sample: {text[:80]!r}")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="fp16,q4,q2,lb0.1,lb0.55,lb1.0")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--init", choices=["svd", "random"], default="svd")
    ap.add_argument("--itq", action="store_true")
    ap.add_argument("--residual", action="store_true")
    ap.add_argument("--label", default="e2e")
    ap.add_argument("--model", default=MODEL, help="HF id (e.g. Qwen/Qwen3-4B)")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"],
                    help="activation/weight dtype for every variant")
    args = ap.parse_args()

    recs = []
    for v in args.variants.split(","):
        # each variant in a fresh process so peak memory is not polluted
        if os.environ.get("_E2E_CHILD"):
            recs.append(run_variant(v, args))
            continue
        cmd = [sys.executable, __file__, "--variants", v, "--tokens", str(args.tokens), "--reps", str(args.reps),
               "--init", args.init, "--label", "_child", "--model", args.model, "--dtype", args.dtype] + (["--itq"] if args.itq else []) + \
              (["--residual"] if args.residual else [])
        out = subprocess.run(cmd, env={**os.environ, "_E2E_CHILD": "1"}, capture_output=True, text=True)
        sys.stdout.write(out.stdout[-2000:])
        if out.returncode != 0:
            sys.stdout.write(out.stderr[-3000:])
            continue
        recs.append(json.loads(out.stdout.strip().splitlines()[-1]))
    if os.environ.get("_E2E_CHILD"):
        print(json.dumps(recs[0]))
        return
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = dict(model=args.model, mlx=mx.__version__, chip=platform.processor(), commit=commit, args=vars(args))
    path = os.path.join(os.path.dirname(__file__), "results", time.strftime("%Y%m%d-%H%M%S") + f"_{args.label}.json")
    with open(path, "w") as f:
        json.dump(dict(meta=meta, records=recs), f, indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
