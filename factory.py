"""unslop-factory: generate an HTML game for every pair of words ("Create 3d pong
game.") with a local model in Unsloth Studio, and collect them in one library page.

    uv run factory.py list                 show every combination and its status
    uv run factory.py generate             make the games that aren't done yet, in random order
    uv run factory.py generate --limit 2   make at most 2 (handy for a first test)
    uv run factory.py generate --only ID   make (or remake, with --force) one game
    uv run factory.py library              rebuild out/library.html

Each machine saves its games in games/<machine-id>/<id>/ (index.html,
meta.json, raw.txt), so several machines can work in one git repo without
conflicts. The machine id comes from local.toml, or else the hostname. After
every game the machine's folder is committed and pushed, and the library is
rebuilt from all machines' games, so you can play while it runs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import socket
import subprocess
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GAMES = ROOT / "games"
OUT = ROOT / "out"
LIBRARY = OUT / "library.html"
TEMPLATE = ROOT / "library_template.html"

# Technical rules only, so the game itself is the model's reading of the words.
SYSTEM_PROMPT = """You are an HTML5 game developer. Every game you make is one complete browser game in a single HTML file:
- Inline CSS and JavaScript only. No external scripts, fonts, images, audio files or network requests. Any sound is generated with the Web Audio API and starts only after the player's first input.
- It runs inside an iframe of any size: fill the window, rescale on resize, no scrollbars.
- A start screen with the game's name and controls, and a way to restart without reloading the page. Listen for keys on window.
- No alert, confirm or prompt. Wrap any localStorage access in try/catch.
- Put the game's name in the <title>.

Reply with only the complete HTML file, in one ```html code block."""


@dataclass
class Combo:
    id: str
    words: tuple[str, ...]
    dir: Path


def slug(text: str, words: int = 4) -> str:
    parts = re.findall(r"[a-z0-9]+", text.lower())
    return "-".join(parts[:words]) or "x"


def load_config() -> dict:
    """config.toml (shared), plus local.toml (this machine only) as cfg["local"]."""
    with open(ROOT / "config.toml", "rb") as f:
        cfg = tomllib.load(f)
    local = ROOT / "local.toml"
    if local.exists():
        with open(local, "rb") as f:
            cfg["local"] = tomllib.load(f)
    return cfg


def machine_id(cfg: dict) -> str:
    return cfg.get("local", {}).get("machine", {}).get("id") or slug(socket.gethostname(), 8)


def combos(cfg: dict) -> list[Combo]:
    p = cfg["permutations"]
    pick = itertools.permutations if p.get("ordered") else itertools.combinations
    machine_dir = GAMES / machine_id(cfg)
    result = []
    for words in pick(p["words"], p.get("words_per_game", 2)):
        cid = "__".join(slug(w) for w in words)
        result.append(Combo(cid, words, machine_dir / cid))
    return result


def prompt_for(cfg: dict, c: Combo) -> str:
    words = list(c.words)
    if not cfg["permutations"].get("ordered"):
        random.shuffle(words)
    return cfg["permutations"]["prompt"].format(words=" ".join(words))


def status(c: Combo) -> str:
    meta = c.dir / "meta.json"
    if not meta.exists():
        return "todo"
    return json.loads(meta.read_text(encoding="utf-8")).get("status", "todo")


# --- Unsloth Studio API ------------------------------------------------------


def api_key(cfg: dict) -> str:
    name = cfg["server"].get("api_key_env", "UNSLOTH_API_KEY")
    if os.environ.get(name):
        return os.environ[name]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == name and value.strip():
                return value.strip().strip('"').strip("'")
    sys.exit(
        f"No API key. Create one in Unsloth Studio (Settings > API keys), then put\n"
        f"  {name}=<key>\nin {env} or set it as an environment variable."
    )


def request(cfg: dict, path: str, body: dict | None = None, timeout: float = 30):
    url = cfg["server"]["base_url"].rstrip("/") + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key(cfg)}",
            "Content-Type": "application/json",
        },
    )
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        if e.code == 401:
            sys.exit(f"Unsloth Studio rejected the API key (401): {detail}")
        raise RuntimeError(f"HTTP {e.code} from {url}: {detail}") from None
    except urllib.error.URLError as e:
        sys.exit(f"Can't reach Unsloth Studio at {url} ({e.reason}). Is it running?")


def pick_model(cfg: dict) -> str:
    with request(cfg, "/models") as r:
        models = [m["id"] for m in json.load(r).get("data", [])]
    wanted = cfg["server"].get("model", "")
    if wanted:
        if wanted not in models:
            print(f"warning: model {wanted!r} not in Studio's list {models}; using it anyway")
        return wanted
    if not models:
        sys.exit("Unsloth Studio has no model loaded. Load Qwen3.8-27B UD-Q8_K_XL in Studio first.")
    return models[0]


def chat(cfg: dict, model: str, prompt: str, on_progress) -> dict:
    """Streams one chat completion. Returns content, reasoning and stats."""
    g = cfg["generation"]
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": g["max_tokens"],
        "temperature": g["temperature"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {
            "enable_thinking": bool(g.get("thinking", False)),
            "reasoning_effort": g.get("reasoning_effort", "xhigh"),
        },
        # Ask Studio not to run its own web-search / code tools.
        "enable_tools": False,
    }
    content, reasoning, finish, usage = [], [], None, {}
    with request(cfg, "/chat/completions", body, timeout=g["timeout_s"]) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                finish = choice.get("finish_reason") or finish
            on_progress(sum(map(len, content)) + sum(map(len, reasoning)))
    return {
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "finish_reason": finish,
        "usage": usage,
    }


def extract_html(text: str) -> str | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"```html\s*\n(.*?)```", text, flags=re.S | re.I)
    html = m.group(1) if m else None
    if html is None:
        m = re.search(r"(<!doctype html.*</html>)", text, flags=re.S | re.I)
        html = m.group(1) if m else None
    if not html or "</html>" not in html.lower() or "<script" not in html.lower():
        return None
    return html.strip() + "\n"


# --- generation ----------------------------------------------------------------

print_lock = threading.Lock()
library_lock = threading.Lock()
git_lock = threading.Lock()


def log(msg: str) -> None:
    with print_lock:
        print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8"
    )


def git_publish(cfg: dict, c: Combo, message: str) -> None:
    """Commit one game's folder, then pull --rebase and push.

    Each machine only writes its own games/<machine-id>/ folder, so the rebase
    never conflicts. A failed push is retried with the next game's push."""
    if not cfg.get("git", {}).get("push", True):
        return
    with git_lock:
        git("add", "--", str(c.dir.relative_to(ROOT)))
        commit = git("commit", "-m", message, "--", str(c.dir.relative_to(ROOT)))
        if commit.returncode != 0 and "nothing" not in commit.stdout + commit.stderr:
            log(f"git commit failed: {commit.stderr.strip() or commit.stdout.strip()}")
            return
        for step in (["pull", "--rebase", "--autostash"], ["push"]):
            res = git(*step)
            if res.returncode != 0:
                log(f"git {step[0]} failed (will retry after the next game): {res.stderr.strip()}")
                return
        log(f"pushed: {message}")


def generate_one(cfg: dict, model: str, c: Combo, n: int, total: int) -> bool:
    prompt = prompt_for(cfg, c)
    c.dir.mkdir(parents=True, exist_ok=True)
    tag = f"[{n}/{total}] \"{prompt}\""
    attempts = 1 + cfg["generation"].get("retries", 0)
    for attempt in range(1, attempts + 1):
        log(f"{tag}: started" + (f" (attempt {attempt})" if attempt > 1 else ""))
        start = time.time()
        last = [start]

        def progress(chars: int) -> None:
            if time.time() - last[0] >= 60:
                last[0] = time.time()
                log(f"{tag}: {chars:,} chars, {time.time() - start:.0f}s")

        try:
            res = chat(cfg, model, prompt, progress)
            error = None
        except Exception as e:  # network drop, timeout, server error
            res, error = {"content": "", "reasoning": "", "finish_reason": None, "usage": {}}, str(e)
        seconds = time.time() - start
        (c.dir / "raw.txt").write_text(
            (f"<think>\n{res['reasoning']}\n</think>\n" if res["reasoning"] else "") + res["content"],
            encoding="utf-8",
        )
        html = None if error else extract_html(res["content"])
        if html is None and not error:
            error = (
                "response was cut off at max_tokens"
                if res["finish_reason"] == "length"
                else "no complete HTML file in the response"
            )
        title = None
        if html:
            m = re.search(r"<title>(.*?)</title>", html, flags=re.S | re.I)
            title = m.group(1).strip() if m else None
            (c.dir / "index.html").write_text(html, encoding="utf-8")
        meta = {
            "id": c.id,
            "machine": machine_id(cfg),
            "words": list(c.words),
            "prompt": prompt,
            "title": title or prompt,
            "status": "done" if html else "failed",
            "error": error,
            "model": model,
            "thinking": cfg["generation"].get("reasoning_effort", "xhigh")
            if cfg["generation"].get("thinking", False)
            else False,
            "attempt": attempt,
            "seconds": round(seconds, 1),
            "finish_reason": res["finish_reason"],
            "completion_tokens": (res["usage"] or {}).get("completion_tokens"),
            "html_bytes": len(html.encode()) if html else 0,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (c.dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if html:
            log(f"{tag}: done in {seconds:.0f}s, \"{meta['title']}\"")
            with library_lock:
                build_library()
            git_publish(cfg, c, f"{meta['machine']}: {meta['title']} ({prompt})")
            return True
        log(f"{tag}: failed: {error}")
    git_publish(cfg, c, f"{meta['machine']}: failed ({prompt}): {error}")
    return False


def build_library() -> int:
    games = []
    for meta_file in sorted(GAMES.glob("*/*/meta.json")):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        page = meta_file.parent / "index.html"
        if meta.get("status") != "done" or not page.exists():
            continue
        meta["html"] = page.read_text(encoding="utf-8")
        meta.setdefault("machine", meta_file.parent.parent.name)
        games.append(meta)
    data = json.dumps(games, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__GAMES__*/[]", data)
    OUT.mkdir(exist_ok=True)
    LIBRARY.write_text(html, encoding="utf-8")
    return len(games)


# --- commands ------------------------------------------------------------------


def cmd_list(cfg: dict, args) -> None:
    all_combos = combos(cfg)
    print(f"Machine: {machine_id(cfg)} ({GAMES / machine_id(cfg)})\n")
    for c in all_combos:
        print(f"{status(c):7} {c.id}")
    counts = {s: sum(status(c) == s for c in all_combos) for s in ("done", "failed", "todo")}
    print(f"\n{len(all_combos)} games: {counts['done']} done, {counts['failed']} failed, {counts['todo']} to do")


def cmd_generate(cfg: dict, args) -> None:
    todo = combos(cfg)
    if args.only:
        todo = [c for c in todo if c.id in args.only]
        missing = set(args.only) - {c.id for c in todo}
        if missing:
            sys.exit(f"Unknown id(s): {', '.join(missing)}. See `factory.py list`.")
    if not args.force:
        todo = [c for c in todo if status(c) != "done"]
    random.shuffle(todo)
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print("Nothing to generate. Use --force to remake games.")
        return
    model = pick_model(cfg)
    parallel = args.parallel or cfg["generation"]["parallel"]
    log(f"Generating {len(todo)} game(s) with {model}, {parallel} at a time, as {machine_id(cfg)}")
    log(f"Library: {LIBRARY}")
    ok = 0
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [
            pool.submit(generate_one, cfg, model, c, i, len(todo))
            for i, c in enumerate(todo, 1)
        ]
        for f in as_completed(futures):
            ok += f.result()
    log(f"Finished: {ok} made, {len(todo) - ok} failed. Library has {build_library()} games.")


def cmd_library(cfg: dict, args) -> None:
    n = build_library()
    print(f"{LIBRARY} ({n} games)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show every combination and its status")
    g = sub.add_parser("generate", help="generate games that aren't done yet")
    g.add_argument("--limit", type=int, help="generate at most this many")
    g.add_argument("--only", nargs="+", metavar="ID", help="only these game ids")
    g.add_argument("--force", action="store_true", help="remake games that are already done")
    g.add_argument("--parallel", type=int, help="override [generation] parallel")
    sub.add_parser("library", help="rebuild out/library.html")
    args = ap.parse_args()
    cfg = load_config()
    {"list": cmd_list, "generate": cmd_generate, "library": cmd_library}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
