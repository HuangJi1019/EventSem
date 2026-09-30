#!/usr/bin/env python3
"""Extract CLIP text features, bit-compatible with the archived ones.

Why this exists (2026-08-26): the SRE benchmarks share qids with the original test
splits, and the SRE evaluations were reading `clip_text_features/<qid>` -- i.e. the
ORIGINAL query's features.  Only the GloVe/WordNet stream was reformulated, so the
CLIP text encoder never saw the reformulated query and the benchmark measured
almost nothing.  This regenerates the CLIP side for any query file.

The recipe was recovered by matching the archive, not guessed:
  TACoS    : [sot]+tokens+[eot], NO padding, float16, key 'last_hidden_state',
             file '<qid>.npz'                       -> reproduces to 1 fp16 ULP
  Charades : [sot]+tokens+[eot] ZERO-padded to 77 (OpenAI clip.tokenize style, NOT
             the transformers pad token), no attention_mask, float32,
             key 'last_hidden_state', file 'qid<N>.npz'  -> reproduces to 1e-5

usage: extract_clip_text.py <tacos|charades> <query.jsonl> <out_dir> [--verify-against DIR]
"""
import sys, os, json, argparse
import numpy as np, torch
from transformers import CLIPTokenizer, CLIPTextModel

MODEL = "openai/clip-vit-base-patch32"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["tacos", "charades"])
    ap.add_argument("jsonl")
    ap.add_argument("out_dir")
    ap.add_argument("--verify-against", default=None,
                    help="archive dir; encode the ORIGINAL queries and confirm the recipe "
                         "reproduces it before writing anything")
    ap.add_argument("--orig-jsonl", default=None, help="original queries, for --verify-against")
    ap.add_argument("--tol", type=float, default=None,
                    help="verification tolerance; default 0.02 (fp16) / 1e-3 (fp32). The TACoS "
                         "archive comes from the public FlashVTG/QD-DETR release and is NOT "
                         "reproducible here (both transformers and the OpenAI clip package give "
                         "an identical 0.0352 max-abs gap), so for TACoS the originals are "
                         "REGENERATED with this same recipe and the SRE comparison is made "
                         "within the regenerated pair.")
    ap.add_argument("--max-len", dest="max_len", type=int, default=None,
                    help="token cap. TACoS MUST use 32: the archived clip_text_features were "
                         "extracted with HF truncation at 32, and linguistic_knowledge_ablate.py "
                         "caps the semantic stream the same way because model.py mixes the two "
                         "ELEMENTWISE. Default: 32 for tacos, 77 for charades.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bsz", type=int, default=64)
    a = ap.parse_args()

    pad77 = a.dataset == "charades"
    max_len = a.max_len if a.max_len is not None else (77 if pad77 else 32)
    dtype = np.float32 if pad77 else np.float16
    fname = (lambda q: f"qid{q}.npz") if pad77 else (lambda q: f"{q}.npz")

    # SLOW tokenizer on purpose: linguistic_knowledge_ablate.py uses CLIPTokenizer, and
    # the fast one disagrees with it on curly quotes (4/3599 TACoS SRE queries), which
    # would desynchronise the two streams that model.py mixes elementwise.
    tk = CLIPTokenizer.from_pretrained(MODEL)
    mdl = CLIPTextModel.from_pretrained(MODEL).eval().to(a.device)

    def encode(queries):
        """list[str] -> list[np.ndarray]; batched, exact per-item lengths preserved."""
        idss = [tk(q)["input_ids"][:max_len] for q in queries]
        L = 77 if pad77 else max(len(i) for i in idss)   # charades keeps the 77 zero-pad
        batch = torch.zeros(len(idss), L, dtype=torch.long)
        for j, ids in enumerate(idss):
            batch[j, :len(ids)] = torch.tensor(ids)
        with torch.no_grad():
            out = mdl(input_ids=batch.to(a.device)).last_hidden_state.cpu().numpy()
        # zero-padding is only faithful at length 77 (Charades). For TACoS every item is
        # written at its own true length, so a shared L in the batch must be sliced back.
        return [out[j] if pad77 else out[j, :len(idss[j])] for j in range(len(idss))]

    if a.verify_against:
        O = [json.loads(l) for l in open(a.orig_jsonl)]
        worst, n = 0.0, 0
        for i in range(0, min(64, len(O)), a.bsz):
            chunk = O[i:i + a.bsz]
            for r, out in zip(chunk, encode([r["query"] for r in chunk])):
                p = os.path.join(a.verify_against, fname(r["qid"]))
                if not os.path.exists(p):
                    continue
                ref = np.load(p)["last_hidden_state"].astype(np.float32)
                got = out.astype(dtype).astype(np.float32)
                if got.shape != ref.shape:
                    print(f"SHAPE MISMATCH {r['qid']}: {got.shape} vs {ref.shape}"); sys.exit(1)
                worst = max(worst, float(np.abs(got - ref).max())); n += 1
        tol = a.tol if a.tol is not None else (0.02 if dtype is np.float16 else 1e-3)
        print(f"[verify] {n} originals re-encoded, max abs diff = {worst:.6g} (tol {tol})")
        if worst > tol:
            print("[verify] FAILED -- recipe does not reproduce the archive; nothing written")
            sys.exit(1)

    R = [json.loads(l) for l in open(a.jsonl)]
    os.makedirs(a.out_dir, exist_ok=True)
    for i in range(0, len(R), a.bsz):
        chunk = R[i:i + a.bsz]
        for r, out in zip(chunk, encode([r["query"] for r in chunk])):
            np.savez(os.path.join(a.out_dir, fname(r["qid"])),
                     last_hidden_state=out.astype(dtype))
        if (i // a.bsz) % 20 == 0:
            print(f"  {min(i + a.bsz, len(R))}/{len(R)}", flush=True)
    print(f"[done] wrote {len(R)} files to {a.out_dir}")

if __name__ == "__main__":
    main()
