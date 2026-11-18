# llama-gauges

An interactive terminal dashboard for [llama.cpp](https://github.com/ggml-org/llama.cpp)'s
`llama-server`. It also works with the llama-server processes that **LM Studio** quietly runs,
one per loaded model.

It shows live generation and prefill speed, slot activity, context fill, speculative / MTP draft
acceptance and a request log, next to GPU, CPU, RAM, PCIe and disk gauges. It is a single
Python file built on [rich](https://github.com/Textualize/rich).

```
 LLAMA   Serving   http://127.0.0.1:53197   Qwen3-8B-Q8_0 32k ×4                         ● Generating · 14%   [LIVE]
 [1 Overview]   2 Slots    3 Requests    4 Hardware
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 Model state   Idle   Reading  [Generating]  Queued   Error                            169 / 1,200 output tokens · 14%
 ████████████████▎━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─  Speed  ────────────────┐ ┌─  GPU load  ─────────────┐ ┌─  VRAM  ─────────────────┐ ┌─  GPU temp  ─────────────┐
│ 71.6 tok/s               │ │ 94 %                     │ │ 12.7 / 16 GB             │ │ 58 °C                    │
│ generating · peak 72.3   │ │ RTX 4070 Ti SUPER        │ │ 80% used · 3.2 GB free   │ │ fan 41% · SM 2610 MHz    │
│ ▁▁▁▁▁▁▁▁▁▁▁▁▁▅▇▇▇▇▇▇▇▇▇▇ │ │ ▁▁▁▁▁▁▁▁▁▁▁▁▁▆██████████ │ │ ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ │ │ ▃▃▃▃▃▃▃▃▃▃▃▃▃▄▅▅▅▅▅▅▅▅▅▅ │
└──────────────────────────┘ └──────────────────────────┘ └──────────────────────────┘ └──────────────────────────┘
┌─  Power  ────────────────┐ ┌─  PCIe  ─────────────────┐ ┌─  CPU  ──────────────────┐ ┌─  Disk read  ────────────┐
│ 139 W                    │ │ Gen4 x16                 │ │ 9 %                      │ │ 0.0 MB/s                 │
│ of 285 W limit           │ │ to GPU 13 · from 216 MB/s│ │ 8 cores · 16 threads     │ │ write 0.1 MB/s           │
│ ▁▁▁▁▁▁▁▁▁▁▁▁▁▄▆▆▆▆▆▆▆▆▆▆ │ │ ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▁▂▁▁▂▁▁▂▁ │ │ ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▁▁▁▁▁▁ │ │ ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ │
└──────────────────────────┘ └──────────────────────────┘ └──────────────────────────┘ └──────────────────────────┘
 Context fill                       355 / 32k  1%   Recent requests
 ▌━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   TIME      STATUS  PROMPT  REUSED  OUTPUT  TOK/S  DURATION
 Draft acceptance     39% · 1.17/draft · 62/34/20   03:09:23  Gen        186      10     169   71.6     2.2 s
 ██████████████████▊━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   03:08:51  Done       271       0     300   65.1     4.1 s
 Slots busy                                 1 / 4   03:08:40  Done        74       2     120   70.1     1.0 s
 ████████████━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 System RAM             15.6 / 30 GB · server 2.0
 █████████████████████████▏━━━━━━━━━━━━━━━━━━━━━━

 1-4 views   Tab model   Space freeze   r refresh   t theme   c clear   ? help   q quit
```

## Features

- **Auto-discovery.** Every `llama-server` process on the machine is found by reading its
  command line: port, model, context size, slots, speculative mode and API key file. Newly
  loaded or reloaded models appear within a few seconds.
- **Live speed.** Tokens/s for generation and prefill come from `/slots` (`n_decoded` and
  prompt-progress deltas), so they move during a request, not only after it.
- **Model state** (Idle / Reading / Generating / Queued / Error) with a progress bar for the
  prompt being read or the output budget.
- **Speculative / MTP decoding**: acceptance rate, accepted tokens per draft, and acceptance by
  draft position.
- **Request log** built from slot task transitions. Output counts and tok/s are corrected with
  the exact `/metrics` counters when a request finishes.
- **Hardware**: GPU load, VRAM, temperature, power against its limit, SM and memory clock, fan,
  PCIe link and throughput in each direction (through NVML), per-thread CPU, RAM, disk read
  and write.
- **Four views, four themes, keyboard driven.** It is read-only and never sends anything to the
  server except GET requests.

| View | Shows |
|---|---|
| **1 Overview** | state row, 8 sparkline tiles, context / draft / slots / RAM bars, recent requests |
| **2 Slots** | slot map and per-slot table: task, prompt, reused, read, output, remaining, tok/s, sampling, context bar |
| **3 Requests** | summary (average tok/s, lifetime tokens, prompt-cache hit rate) and the full request log |
| **4 Hardware** | long-history sparklines for every sensor, per-thread CPU bars (works with no server) |

| Key | Action |
|---|---|
| `1`–`4` | switch view |
| `Tab` / `←` `→` | switch between loaded models |
| `Space` | freeze / unfreeze |
| `r` | poll now |
| `t` | cycle theme (ocean, ember, forest, mono) |
| `c` | clear history, peaks and the request log |
| `?` | help |
| `q` | quit (servers keep running) |

## Install

Requires Python 3.9+.

```bash
pip install -r requirements.txt          # rich, psutil, nvidia-ml-py (optional)
python llama_gauges.py
```

or install it as a command:

```bash
pip install .[nvidia]
llama-gauges
```

`nvidia-ml-py` is optional. Without it the GPU tiles fall back to `nvidia-smi`, without PCIe
throughput. On machines without an NVIDIA GPU the GPU tiles are hidden.

## Server setup

llama-gauges needs two llama-server endpoints:

| Endpoint | Default | Needed for |
|---|---|---|
| `/metrics` | **off**, enable with `--metrics` | counters: lifetime tokens, draft acceptance, queue |
| `/slots` | on (unless `--no-slots`) | live speed, state, progress, slot table, request log |

Both sit behind the server's API key when it has one.

### Plain llama-server

```bash
llama-server -m model.gguf --port 8080 --metrics
# with a key:
llama-server -m model.gguf --port 8080 --metrics --api-key-file my-key.txt
```

The process is discovered automatically. For a server on another machine, or one whose
process you can't see, use `--port` and `--key-file`.

### LM Studio

LM Studio's llama.cpp engine starts a separate `llama-server` for each loaded model. It runs
on a random `127.0.0.1` port with a random per-load API key. LM Studio's own API (`lms server`,
port 1234) does **not** pass `/metrics` or `/slots` through.

LM Studio lets you append extra llama-server arguments to a model's load config. Two flags are
enough:

```
--metrics --api-key-file <path-to>/metrics-key.txt
```

`--api-key-file` **adds** your key next to LM Studio's own key, so LM Studio keeps working and
llama-gauges reads the key from that file. The helper script writes the flags into every
installed LLM:

```bash
python tools/lmstudio_flags.py --create-key     # creates <lmstudio-dir>/metrics-key.txt, adds flags
python tools/lmstudio_flags.py --list           # show current overrides per model
python tools/lmstudio_flags.py --dry-run        # preview
python tools/lmstudio_flags.py --remove         # undo
```

Then reload the model(s) in LM Studio. Notes:

- Regular models get the flags in `.internal/user-concrete-model-default-config/<model>.gguf.json`.
  Hub models (`publisher/model`) ignore that file for load settings, so for them the script
  edits `config.load` in `hub/models/<publisher>/<model>/model.yaml`.
- Each changed file is backed up once as `*.bak-llama-gauges`.
- If LM Studio re-downloads a hub `model.yaml`, or another tool regenerates it, the flags are
  lost. Run the script again.
- The LM Studio dir defaults to `~/.lmstudio`. Override it with `--lmstudio-dir` or `LMSTUDIO_HOME`.
- Embedding and reranker models are skipped unless you pass `--include-embeddings`.

If you edit the flags by hand in LM Studio's UI, watch for **stray spaces**. A key saved as
`" --api-key-file"` (leading space) makes llama-server exit with
`error: invalid argument:  --api-key-file` (note the two spaces). The helper script strips them.

## Usage

```
python llama_gauges.py [options]

  -i, --interval SEC     poll interval (default 1)
  -p, --port PORT        only watch this port (repeatable); also adds servers not found as processes
  -k, --key-file FILE    API key file (overrides the server's --api-key-file)
  --view N               start view 1-4
  --theme NAME           ocean | ember | forest | mono
  --no-screen            draw inline instead of the full-screen buffer
  --no-gpu               don't read GPU sensors
  --once                 print one snapshot and exit (handy for checking a setup)
  --version
```

## How the numbers are made

- **Live tok/s** is the sum, over busy slots, of the change in `n_decoded` between two polls,
  divided by the time between them, smoothed with an EMA. Prefill uses
  `n_prompt_tokens_processed` the same way. When idle, the Speed tile shows the last finished
  request.
- **`/metrics` counters** (`tokens_predicted_total` and so on) only change when a prefill or
  generation finishes. llama-gauges uses them for lifetime totals, draft acceptance and exact
  per-request numbers. It computes its own deltas from the `*_total` counters. The rate gauges
  (`prompt_tokens_seconds`, `predicted_tokens_seconds`) cover only the window since the previous
  scrape, so they aren't used.
- **Request log**: a request appears when a slot picks up a new task and is closed when the
  task disappears. Duration is accurate to about one poll interval. Output and tok/s are exact
  when one request finishes per poll. History only covers the time the dashboard was running.
- **Draft acceptance** = `spec_decode_num_accepted_tokens_total / spec_decode_num_draft_tokens_total`.
  "/draft" means accepted tokens per draft, and the per-position numbers are the share of drafts
  whose token at position *n* was accepted.

## Troubleshooting

| Panel says | Fix |
|---|---|
| `401 unauthorized` | the server has an API key: add `--api-key-file <file>` to its flags, or pass `--key-file` |
| `501 metrics disabled` | add `--metrics` |
| `/slots HTTP 501` | the server runs with `--no-slots`, so live speed and the slot views are unavailable |
| `No llama-server process found` | no process named `llama-server*` with a `--port` argument; use `--port` for remote servers |
| boxes or `?` instead of bars | use a terminal and font with Unicode block characters (Windows Terminal with Cascadia works) |

## Security notes

- `/slots` can include prompt text and sampling parameters. llama-gauges displays only
  counters and sampling values, never prompt content.
- The API key file is a plain secret. Keep it outside any repo (`*key*.txt` is git-ignored here).
- llama-gauges never reads the server's own `--api-key` from the process command line. It only
  uses the key file.

## Tests

```bash
python -m unittest discover -s tests
```

## License

MIT
