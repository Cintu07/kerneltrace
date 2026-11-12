"""does the softmax-invariant key bias show up outside pythia?

the invariant part is whatever of a head's key bias rotary leaves alone: all of it in models with learned
or alibi positions (gpt-2, opt, bloom, cerebras-gpt, santacoder), the non-rotary dims in partial-rotary
models (gpt-neox-20b, stablelm-alpha, phi), and nothing in full-rotary models (qwen2.5), which are here as
the contrast. reads only config.json and the bias tensors, with http range requests, no weights downloaded.

the number per model is the largest per-head norm of the invariant part over all layers, the same measure
as the pythia suite in scripts/key_bias.py. the query bias is there for scale.

usage: python scripts/key_bias_families.py [repo ...]
writes evidence/key_bias_families/families.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from key_bias import _remote_tensors  # noqa: E402

OUT = Path("evidence/key_bias_families/families.json")
MODELS = [
    # learned positions, so the whole key bias is invariant
    "openai-community/gpt2", "openai-community/gpt2-medium", "openai-community/gpt2-large", "openai-community/gpt2-xl",
    "facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b", "facebook/opt-6.7b",
    "cerebras/Cerebras-GPT-111M", "cerebras/Cerebras-GPT-590M", "cerebras/Cerebras-GPT-1.3B", "cerebras/Cerebras-GPT-2.7B",
    "bigcode/gpt_bigcode-santacoder",
    # alibi
    "bigscience/bloom-560m", "bigscience/bloom-1b1", "bigscience/bloom-1b7", "bigscience/bloom-3b", "bigscience/bloom-7b1",
    # partial rotary
    "EleutherAI/gpt-neox-20b", "stabilityai/stablelm-base-alpha-3b", "stabilityai/stablelm-base-alpha-7b",
    "microsoft/phi-1_5", "microsoft/phi-2",
    # full rotary: the key bias is visible to the loss
    "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B",
]


def config(repo, rev):
    from huggingface_hub import hf_hub_download

    return json.loads(Path(hf_hub_download(repo, "config.json", revision=rev)).read_text(encoding="utf-8"))


def tensor_names(repo, rev):
    """every tensor name in the checkpoint, from the safetensors index or the single file's header."""
    import struct

    import requests
    from huggingface_hub import hf_hub_url

    idx = requests.get(hf_hub_url(repo, "model.safetensors.index.json", revision=rev), timeout=60)
    if idx.ok:
        return list(idx.json()["weight_map"])
    url = hf_hub_url(repo, "model.safetensors", revision=rev)
    r = requests.get(url, headers={"Range": "bytes=0-7"}, timeout=60)
    if not r.ok:
        raise FileNotFoundError(f"no safetensors ({r.status_code})")
    n = struct.unpack("<Q", r.content)[0]
    head = requests.get(url, headers={"Range": f"bytes=8-{8 + n - 1}"}, timeout=60).json()
    return [k for k in head if k != "__metadata__"]


def layer_of(name):
    m = re.search(r"\.(\d+)\.", name)
    return int(m.group(1))


def safetensors_revision(repo):
    """main if it has safetensors, else a pull-request ref where the hub's bot converted the same weights."""
    from huggingface_hub import list_repo_files, list_repo_refs, model_info

    main = model_info(repo).sha
    files = list_repo_files(repo, revision=main)
    if any(f.endswith(".safetensors") for f in files):
        return main, "main"
    for pr in list_repo_refs(repo, include_pull_requests=True).pull_requests or []:
        if any(f.endswith(".safetensors") for f in list_repo_files(repo, revision=pr.ref)):
            return pr.target_commit, pr.ref
    return main, "main"


def survey(repo):
    rev, ref = safetensors_revision(repo)
    cfg = config(repo, rev)
    arch = cfg.get("model_type")
    names = tensor_names(repo, rev)
    d = cfg.get("hidden_size") or cfg.get("n_embd") or cfg.get("n_embed") or cfg.get("d_model")
    heads = cfg.get("num_attention_heads") or cfg.get("n_head")
    hd = d // heads
    rot, scheme = 0, "learned positions"
    if arch == "gpt_neox":
        rot, scheme = int(hd * cfg.get("rotary_pct", 1.0)), "partial rotary"
    elif arch == "phi":
        rot, scheme = int(hd * cfg.get("partial_rotary_factor", 1.0)), "partial rotary"
    elif arch == "bloom":
        scheme = "alibi"
    elif arch == "qwen2":
        rot, scheme = hd, "full rotary"
    elif arch == "stablelm":
        rot, scheme = int(hd * cfg.get("partial_rotary_factor", 0.25)), "partial rotary"

    pat = {"gpt2": r"attn\.c_attn\.bias$", "gpt_bigcode": r"attn\.c_attn\.bias$",
           "opt": r"self_attn\.(k|q)_proj\.bias$", "bloom": r"self_attention\.query_key_value\.bias$",
           "gpt_neox": r"attention\.query_key_value\.bias$", "phi": r"self_attn\.(k|q)_proj\.bias$",
           "stablelm": r"self_attn\.(k|q)_proj\.bias$", "qwen2": r"self_attn\.(k|q)_proj\.bias$"}[arch]
    wanted = sorted((n for n in names if re.search(pat, n)), key=lambda n: (layer_of(n), n))
    if not wanted:
        return {"repo": repo, "revision": rev, "arch": arch, "scheme": scheme, "error": "no attention bias tensors"}
    t = _remote_tensors(repo, rev, wanted)
    kv_heads = cfg.get("num_key_value_heads") or heads
    layers = {}
    for name, x in t.items():
        L = layers.setdefault(layer_of(name), {})
        if arch == "gpt2":
            L["q"], L["k"] = x[:d].view(heads, hd), x[d:2 * d].view(heads, hd)
        elif arch == "gpt_bigcode":   # multi-query: one shared key head
            L["q"], L["k"] = x[:d].view(heads, hd), x[d:d + hd].view(1, hd)
        elif arch in ("bloom", "gpt_neox"):
            b = x.view(heads, 3, hd)
            L["q"], L["k"] = b[:, 0], b[:, 1]
        elif ".k_proj." in name:
            L["k"] = x.view(kv_heads, hd)
        elif ".q_proj." in name:
            L["q"] = x.view(heads, hd)
    inv = [float(L["k"][:, rot:].norm(dim=-1).max()) if rot < hd else 0.0 for L in layers.values()]
    vis = [float(L["k"][:, :rot].norm(dim=-1).max()) if rot else float("nan") for L in layers.values()]
    qb = [float(L["q"].norm(dim=-1).max()) for L in layers.values() if "q" in L]
    worst = max(range(len(inv)), key=lambda i: inv[i])
    slow = None
    if scheme == "full rotary":
        # rotate_half pairs dim i with i + hd/2 at frequency base^(-2i/hd). a pair that turns by under 0.1 rad
        # across 4096 positions is close to invariant: its part of q.b is nearly the same for every key.
        base = float(cfg.get("rope_theta", 10000.0))
        inv_freq = base ** (-torch.arange(0, hd, 2).float() / hd)
        still = (inv_freq * 4096 < 0.1)
        idx = torch.cat([still.nonzero().flatten(), still.nonzero().flatten() + hd // 2])
        fast = torch.cat([(~still).nonzero().flatten(), (~still).nonzero().flatten() + hd // 2])
        slow = {"rope_theta": base, "near_invariant_dims": int(idx.numel()),
                "near_invariant_max": max(float(L["k"][:, idx].norm(dim=-1).max()) for L in layers.values()),
                "turning_max": max(float(L["k"][:, fast].norm(dim=-1).max()) for L in layers.values())}
    return {"repo": repo, "revision": rev, "ref": ref, "arch": arch, "scheme": scheme, "layers": len(layers), "heads": heads,
            "head_dim": hd, "rotary_dims": rot, "invariant_max": max(inv), "invariant_layer": worst,
            "invariant_per_layer": inv, "rotary_part_max": max(vis) if rot else None,
            "query_bias_max": max(qb) if qb else None, "dtype_on_hub": cfg.get("torch_dtype"), "full_rotary_split": slow}


def main(repos):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for repo in repos:
        try:
            r = survey(repo)
        except Exception as e:  # gated, missing safetensors, network
            r = {"repo": repo, "error": f"{type(e).__name__}: {e}"[:200]}
        result[repo] = r
        if "error" in r:
            print(f"{repo:40s} skipped: {r['error']}", flush=True)
        else:
            rp = "" if r["rotary_part_max"] is None else f"  rotary part {r['rotary_part_max']:7.1f}"
            print(f"{repo:40s} {r['scheme']:17s} invariant {r['invariant_max']:8.1f} (layer {r['invariant_layer']:2d})"
                  f"{rp}  query bias {r['query_bias_max'] or float('nan'):6.1f}", flush=True)
            if r.get("full_rotary_split"):
                s = r["full_rotary_split"]
                print(f"{'':40s} {s['near_invariant_dims']} of {r['head_dim']} dims turn < 0.1 rad over 4096 tokens: "
                      f"bias there {s['near_invariant_max']:.1f}, in the rest {s['turning_max']:.1f}", flush=True)
        OUT.write_text(json.dumps(result, indent=1), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:] or MODELS)
