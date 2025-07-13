#!/usr/bin/env python3
"""
ai_cli.py  ·  directory-aware chat client for OpenRouter or OpenAI

New flags
---------
--include-glob "*.py,*.md"      # only visit matching paths   (default: **/*)
--exclude-glob "data/**,*.gz"   # skip matching paths         (default: none)
--stop-after  N                 # hard-stop after N files     (default: 0 = no limit)
--preview-len N                 # chars to print per reply    (default: 200)

Install deps:
    pip install requests tiktoken tqdm
"""

import argparse, os, json, pathlib, fnmatch
from collections import defaultdict
import requests, tiktoken
from tqdm import tqdm

# ------------------------------------------------------------
# 1. Pricing – (prompt_cost, completion_cost) in USD / 1 000 tokens
# ------------------------------------------------------------
PRICES_USD_PER_1K = {
    "openai/chatgpt-4o-latest"     : (0.005 , 0.015),
    "openai/gpt-4.1"               : (0.002 , 0.008),
    "openai/o3"                    : (0.002 , 0.008),
    "anthropic/claude-opus-4"      : (0.015 , 0.075),
    "google/gemini-2.5-pro"        : (0.00125, 0.010),
    "mistralai/mistral-large-2411" : (0.002 , 0.006),
    "deepseek/deepseek-r1-0528"    : (0.0005, 0.00215),
    "qwen/qwen3-8b"                : (0.000035,0.000138)
}

# ------------------------------------------------------------
# 2. OpenRouter slug ➜ native OpenAI model id (for --provider openai)
# ------------------------------------------------------------
OPENAI_MODEL_MAP = {
    "openai/chatgpt-4o-latest": "gpt-4o-2024-08-06",
    "openai/gpt-4.1"          : "gpt-4.1-2025-04-14",
    "openai/o3"               : "o3-2025-04-16",
}

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def get_encoder(model: str):
    """Fallback-safe tokenizer getter."""
    try:
        return tiktoken.encoding_for_model(model.split("/")[-1])
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")

def count_tokens(enc, text: str) -> int:
    try:
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 3)          # crude fallback

def path_matches(path: pathlib.Path, patterns):
    return any(fnmatch.fnmatch(str(path), pat) for pat in patterns)

def yield_files(root: pathlib.Path, include_glob: str, exclude_glob: str):
    inc = [p.strip() or "**/*" for p in include_glob.split(",")] if include_glob else ["**/*"]
    exc = [p.strip()           for p in exclude_glob.split(",")] if exclude_glob else []
    skip_dirs = {".git", "node_modules", ".cache", "__pycache__"}

    for p in root.rglob("*"):
        if p.is_dir() and p.name in skip_dirs:
            continue
        if not p.is_file():
            continue
        if path_matches(p, exc):
            continue
        if not path_matches(p, inc):
            continue
        if p.stat().st_size > 512_000:          # 512 KB hard limit
            continue
        yield p

def chunk_source(src: str, max_tokens: int, enc):
    """Yield token-bounded chunks."""
    if count_tokens(enc, src) <= max_tokens:
        yield src
        return
    buf = []
    for line in src.splitlines():
        buf.append(line)
        if count_tokens(enc, "\n".join(buf)) >= max_tokens:
            buf.pop()
            yield "\n".join(buf)
            buf = [line]
    if buf:
        yield "\n".join(buf)

def request_chat(provider: str, model: str, system_prompt: str, user_prompt: str):
    if provider == "openrouter":
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "usage": {"include": True},
        }
    else:  # openai direct
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json",
        }
        native = OPENAI_MODEL_MAP.get(model, model.split("/", 1)[-1])
        payload = {
            "model": native,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        }

    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    r.raise_for_status()
    return r.json()

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["openrouter", "openai"], default="openrouter")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dir", default=".")
    ap.add_argument("--prompt", default="Summarise this file:")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--include-glob", default="")
    ap.add_argument("--exclude-glob", default="")
    ap.add_argument("--stop-after", type=int, default=0)
    ap.add_argument("--preview-len", type=int, default=200)
    args = ap.parse_args()

    enc = get_encoder(args.model)
    sys_prompt = "You are a concise, code-aware assistant."
    totals = defaultdict(float)
    completed_files = 0

    root = pathlib.Path(args.dir)
    files = list(yield_files(root, args.include_glob, args.exclude_glob))

    for path in tqdm(files, desc="Processing files"):
        src = path.read_text(errors="ignore")
        for i, chunk in enumerate(chunk_source(src, args.max_tokens, enc)):
            user_msg = f"""{args.prompt}

File: {path}{'' if i==0 else f' (chunk {i+1})'}

```text
{chunk}
```"""
            data = request_chat(args.provider, args.model, sys_prompt, user_msg)

            # ----- usage accounting -----
            if args.provider == "openrouter":
                u = data["usage"]
                p_tok, c_tok, usd = u["prompt_tokens"], u["completion_tokens"], u["cost"]
            else:
                u = data["usage"]
                p_tok, c_tok = u["prompt_tokens"], u["completion_tokens"]
                in_cost, out_cost = PRICES_USD_PER_1K.get(args.model, (0, 0))
                usd = (p_tok / 1000) * in_cost + (c_tok / 1000) * out_cost

            totals["prompt"]     += p_tok
            totals["completion"] += c_tok
            totals["total"]      += p_tok + c_tok
            totals["usd"]        += usd

            reply = data["choices"][0]["message"]["content"].strip()
            full = reply
            if args.preview_len and len(full) > args.preview_len:
                preview = full[:args.preview_len] + "…"
            else:
                preview = full
            print(f"\n[{path.name}{f':{i+1}' if i else ''}] {preview}\n")

        completed_files += 1
        if args.stop_after and completed_files >= args.stop_after:
            break

    # ----- summary -----
    print("\n----- SUMMARY -----")
    print(f"Prompt tokens   : {int(totals['prompt'])}")
    print(f"Completion tok. : {int(totals['completion'])}")
    print(f"Total tokens    : {int(totals['total'])}")
    print(f"Estimated cost  : ${totals['usd']:.4f}")

if __name__ == "__main__":
    main()