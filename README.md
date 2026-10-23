# unslop-factory

Generates a browser game for every pair of words in [config.toml](config.toml),
prompting a local model in Unsloth Studio with just `Create {word1} {word2} game.`,
and collects them in one library page to play them.

Several machines can run it at once. Each one saves its games in
`games/<machine-id>/` and commits and pushes after every game.

## Set up a machine

1. Install [uv](https://docs.astral.sh/uv/) and Git for Windows, and start
   Unsloth Studio with the model loaded (Qwen3.8-27B-GGUF, UD-Q8_K_XL).
2. Clone the repo:
   ```powershell
   git clone https://inetslop@github.com/inetslop/unslop-factory.git
   cd unslop-factory
   ```
3. Copy `.env.example` to `.env` and paste an API key from Unsloth Studio.
4. Copy `local.example.toml` to `local.toml` and set the machine id (the
   default is the hostname). Use the same id as in ai-bench.

## Run

```powershell
uv run factory.py list                 # this machine's games and their status
uv run factory.py generate --limit 1   # one random game
uv run factory.py generate             # all games this machine hasn't made yet
uv run factory.py library              # rebuild out/library.html
```

Open `out/library.html` to play. It's rebuilt after every game and holds the
games of every machine in `games/` (`git pull` to get the others').

## Layout

```
games/<machine-id>/<word1>__<word2>/
  index.html   the game
  meta.json    prompt, title, model, thinking level, time, tokens
  raw.txt      the model's full reply, including its thinking
```
