# AGENTS.md — Hinglish Turn Detection Project

## What this project is

Building a **Hinglish (Hindi-English code-mixed) turn detection model** for the Shiprocket
Data Scientist challenge. The model answers one binary question about a speech segment:
*is the user done speaking (end of turn) or just pausing?*

- **Reference baseline**: Pipecat Smart Turn v3.2 (`smart_turn_reference/` — read-only reference copy, do not modify).
- **Our work happens in `turn_v2/`** and root-level scripts.
- **Operational masterplan**: `context/masterplan.md` — phases, target `turn_v2/` layout,
  training recipe, exit criteria. **Read this first**, then the playbook below.
- **Strategy playbook**: `context/references/Hinglish-Turn-Detection-Playbook.md` — the 7-day
  roadmap, data budget, and upgrade priorities. Read the section relevant to a task before
  proposing plans.
- **Baseline internals**: `context/references/CODE_REPORT.md` — Smart Turn architecture,
  training/quantization pipeline, known flaws.

## Repository layout

```
pipe_cat_reference/
├── AGENTS.md                  # this file
├── pyproject.toml             # uv-managed deps
├── .env                       # SECRETS - never read contents into context, never commit
├── smart_turn_reference/                # Pipecat reference implementation (READ-ONLY)
├── turn_v2/                   # OUR model code goes here (currently empty)
├── scripts/                   # data ops (download, TTS generation, dataset restructuring)
├── data/                      # downloaded/generated datasets (gitignored if repo becomes git)
└── context/                   # planning & progress system (see "Context workflow" below)
    ├── masterplan.md          # operational plan: phases, layout, recipe, exit criteria
    ├── phases/                # phase plans (phase1.md, phase2.md, ...) — progress trace
    ├── subplan.md             # detailed task breakdown of the CURRENT phase (mutable)
    ├── HARDPOINT.md           # log of stuck points / blockers
    ├── progress.md            # session-by-session progress log
    └── references/            # playbook, code report, training reference, sample audio
```

## Environment & package management

- **Package manager: `uv`. Do NOT use pip directly.**
- venv lives at `.venv/` (Python 3.11).

```powershell
uv sync                                  # install/sync deps from lockfile
uv add <pkg>                             # add a dependency (updates pyproject + lock)
uv run python script.py                  # run anything inside the env
uv run python -c "import datasets"       # quick sanity check
```

- Heavy ML deps (torch/torchaudio GPU builds, transformers, onnxruntime, onnx) are NOT yet
  installed — add them with `uv add` only when a task needs them.
- On this machine (RTX 4060 8GB): prefer CPU wheels unless CUDA needed; torch CUDA via uv:
  `uv add torch --index https://download.pytorch.org/whl/cu124` style index pinning if required.

## Secrets

- `.env` contains `SARVAM_API_KEY` and `HUGGINGFACE_API_KEY`.
- **Never print/read .env file contents into the conversation or logs.**
- Load at runtime: `from dotenv import load_dotenv; load_dotenv()`.
- HF auth when needed: `uv run huggingface-cli login` or set `HF_TOKEN` from the env var in-process.

## Data strategy (the 1–2 GB cap — respect it)

Full `pipecat-ai/smart-turn-data-v3.2-train` is **41.4 GB / 270k rows — never bulk-download it.**
Subset targets per playbook Part 5.2 (~1.9 GB total):

| Source | Quota | Purpose |
|---|---|---|
| v3.2-train `hin` | 6,000 | base supervised signal |
| v3.2-train `eng` | 2,000 | code-switch anchor |
| v3.2-train `mar`+`ben` | 400+400 | Indic prosody diversity — on disk but EXCLUDED from training (pivot) |
| TTS Hinglish (edge-tts) | ~232 existing clips (scale-up CANCELLED, pivot 2026-08-29) | Stage-2 finetune domain |
| v3.2-test hin+eng subset | ~1,200 | test-A in-distribution eval |
| MUCS 2021 subset | 1–2 hrs | test-C realism eval |

**Training recipe is two-stage** (masterplan §4): Stage 1 pretrain on hin+eng only →
Stage 2 finetune on TTS Hinglish + 50:50 hin/eng replay (lr 1e-5, 2 epochs).

Subsampling rules: keep 50:50 complete/incomplete per language; preserve midfiller/endfiller
diversity; prefer shorter clips. Data lands under `data/` (gitignored if repo becomes git).

## Conventions

- Input contract everywhere (must match Smart Turn exactly):
  16 kHz mono float32 [-1,1]; ≤8 s; shorter → zero-pad FRONT; longer → keep LAST 8 s
  (`truncate_audio_to_last_n_seconds()` semantics — see `smart_turn_reference/audio_utils.py`).
- Labels: `endpoint_bool` (1 = complete = agent should respond), plus `midfiller`/`endfiller`.
- Keep 200 ms trailing silence convention for any generated clips; never cut mid-word.
- Every experiment must produce numbers on all three test sets (A: official subset,
  B: held-out TTS Hinglish, C: MUCS real speech). One change per ablation row.

## Commands cheat-sheet

```powershell
uv sync                                        # restore env
uv run python scripts/download_data.py         # subset download (streaming filter)
uv run python smart_turn_reference/predict.py <wav>      # baseline inference on a file (needs onnxruntime)
uv run python smart_turn_reference/record_and_predict.py # live mic demo (needs pyaudio)
```

## Context workflow (how planning documents are used)

Work is planned and traced through `context/`. Follow this order on every task:

1. **Read `context/masterplan.md` first.** It defines the phases (P0–P7), the `turn_v2/`
   layout, the training recipe, and each phase's exit criteria.
2. **Check `context/phases/` for the current phase plan.** Phase plans are created
   **one at a time** (`phase1.md`, `phase2.md`, …) — write the next phase plan ONLY after
   the current phase is complete (its exit criteria met and logged). The `phases/` folder
   doubles as the project progress trace: completed phases stay there, never deleted.
3. **Write `context/subplan.md` for the current phase.** The subplan breaks the phase into
   **4–5 subprocesses**, each with **3–6 concrete tasks**. The subplan is **mutable** —
   it is rewritten at the start of each phase and updated (task checkboxes) as work proceeds.
   Only one subplan exists at a time (the current phase's).
4. **Log blockers in `context/HARDPOINT.md`.** Any step where you get stuck (error, unclear
   spec, failing verification) gets a dated entry: what was attempted, what failed, current
   workaround or open question. Append, don't overwrite — it is a history, not a status board.
5. **Session state goes to `context/progress.md`** — one dated section per session: done,
   pending, key decisions.

Reading order for a new agent: `AGENTS.md` → `context/masterplan.md` → latest
`context/phases/phaseN.md` → `context/subplan.md` → `context/HARDPOINT.md` (last entries).

## Working rules for agents

1. Read the playbook section relevant to a task BEFORE implementing it.
2. Don't modify anything under `smart_turn_reference/` — import from it instead.
3. New code goes in `turn_v2/` or root `scripts/`; keep modules small and runnable.
4. Verify audio stats after any data step (counts per language × label, durations, disk size).
5. Prefer generating tables/curves over prose when reporting results (playbook rule).
6. Timebox per playbook Day plan; if blocked >1 session on an upgrade arm (e.g. distillation),
   fall back to shipping the best ablation-arm model.


## Context

The `context/` folder is the single source of planning truth for agents:
`masterplan.md` (what/why) → `phases/` (when, one phase at a time) →
`subplan.md` (how, current phase only) → `HARDPOINT.md` (what's stuck) →
`progress.md` (what happened). Keep each file current; the next agent depends on it.