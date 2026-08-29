# Zero to Hero: Building a Hinglish Turn Detection Model
### A complete playbook for the Shiprocket Data Scientist challenge

> **Constraints for this plan:** ~1 week timeline · local RTX 4060 (8GB VRAM) · **dataset storage cap of 1–2GB** · data sources: Pipecat Smart Turn data (subset), TTS-synthesized Hinglish, public Indic corpora.
>
> Goal: build a **tiny, fast, accurate, audio-native turn detection model** for Indian Hinglish speech — and, more importantly, demonstrate senior-level thinking about data, experiments, and trade-offs.
>
> Deliverables: Hugging Face / Gradio demo **or** GitHub repo with weights + run instructions, plus a self-written report.

---

## Part 1 — What This Challenge Is Actually Testing

The brief says it plainly: *"The solution doesn't need to be perfect. I care more about your data preparation, experiments, depth of solutioning and approach."*

They are hiring someone to own turn detection in production. So the evaluation rubric is roughly:

| What they look for | How you show it |
|---|---|
| Do you understand the problem deeply? | Report section on why turn detection is hard, why Hinglish specifically |
| Can you handle messy data? | Documented dataset audit: label noise, balance, edge cases, listening notes |
| Can you build a working model? | Trained model, weights, reproducible training code |
| Do you evaluate like an engineer? | Latency-vs-accuracy curves, per-category metrics, error analysis, ablations |
| Do you know the field? | Positioning against Smart Turn v3, TEN VAD, LiveKit turn detector |
| Do you understand your own code? | Human-written report, clear README, honest limitations |

Every choice below is optimized for that rubric, not for leaderboard accuracy.

---

## Part 2 — Background: Turn Detection, Properly Understood

### 2.1 The problem

A voice agent must decide, on every pause: **is the user done speaking (end of turn) or just pausing (turn continuation)?**

- Predict "done" too eagerly → the agent **interrupts** the user (catastrophic UX).
- Wait too long → **dead air**, the conversation feels robotic.

This decision must happen continuously, in real time, under uncertainty. It is widely considered one of the hardest open problems in voice AI infrastructure.

### 2.2 How humans solve it — the three cue families

Humans do not use silence duration. We use:

1. **Semantic content** — is the utterance syntactically/pragmatically complete? ("Tomorrow" is a complete answer *only* given the question's context.)
2. **Context** — the dialogue history, what the agent just asked.
3. **Prosody** — intonation, pitch contour, energy, rhythm. Rising pitch = holding the floor; falling pitch = completion.

The history of turn detection is the story of which cues you can capture, at what latency, at what cost.

### 2.3 The taxonomy of approaches in production (mid-2026)

**Tier 0 — Silence-based VAD.** Silero VAD / TEN VAD detect speech vs. non-speech; "end of turn" = silence for N ms (typically 500–1000ms). Still the production default in many systems. Fails exactly on contemplative pauses, fillers, and mid-list pauses (dictating phone numbers, addresses — extremely common in Indian e-commerce/support calls). **This is your baseline to beat.**

**Tier 1 — Text/semantic turn detection.** Feed the streaming transcript into a small LM that predicts completion (TurnGPT; LiveKit's original 135M EOU model; their later Qwen2.5-0.5B multilingual distill; TEN Turn Detection). Structural ceiling: ASR errors propagate, transcription adds latency, and **text discards prosody** — "I would like to order one large pizza…" is textually identical whether done or continuing. ASRs also often strip fillers ("um", "hmm") that are strong turn-holding signals.

**Tier 2 — Audio-native classifiers.** Classify the raw waveform. **Pipecat Smart Turn v3** (the reference implementation for this challenge): Whisper Tiny encoder + classification head, ~8M active params, 8MB int8 ONNX, ~10–100ms CPU inference, 23 languages including Hindi. Runs only during silence, gated by a VAD. **Academically, VAP (Voice Activity Projection)** predicts future voice activity of both speakers from raw stereo audio, self-supervised — handles turn-shifts, backchannels, and interruptions uniformly.

**Tier 3 — Fused semantic + acoustic.** LiveKit Turn Detector v1 (2026): an audio encoder projects into an LLM's embedding space (semantic branch, no transcript) plus a separate acoustic branch for prosody, fused into one prediction. They also released **eot-bench**, the first open end-of-turn benchmark — and it evaluates *full endpointing policies*, not raw model scores.

**Tier 4 — Turn detection absorbed into the STT or the speech model.** Deepgram Flux builds turn detection into recognition (~260ms P50 end-of-turn latency, with "eager end-of-turn" events for speculative LLM inference); OpenAI Realtime's `semantic_vad` and Gemini Live's automatic activity detection make it an internal capability of speech-to-speech models.

**Tier 5 — The policy layer.** Whatever the model outputs, production wraps it in a decision policy: min/max endpointing delays, confidence thresholds (per-language), eager/speculative response triggers, timeout fallbacks, VAD gating. **The model outputs a probability; the policy turns it into behavior; the latency-vs-false-cutoff Pareto frontier of that policy is what users actually experience.** This is the key maturity insight — bake it into your evaluation.

### 2.4 Why Hinglish is a genuinely hard, unsolved case

This is your differentiator — the reason Shiprocket is hiring for this:

- **Code-switching breaks text-based approaches structurally.** Monolingual ASRs commit to one language per utterance; mid-sentence Hindi↔English switches cause substitution/deletion errors, and any text-based EOT model downstream inherits garbage. On noisy code-switched telephony audio, global ASR models hit 14–16% WER vs 9–11% for Indic-specialized models. **An audio-native model never needs a transcript — it sidesteps this failure mode entirely.**
- **Fillers carry signal but ASR drops them.** "umm", "matlab", "actually", "toh" are turn-holding cues that vanish in transcripts. Audio-native keeps them.
- **Indian prosody and pause conventions differ** from the English data most models were trained on — pause durations, clause-boundary intonation. VAD defaults tuned on English systematically mis-handle other cadences.
- **Romanization ambiguity** ("the same Hindi word in multiple spellings") makes text features fragile.
- The best open multilingual models (Smart Turn v3, LiveKit v1) cover *Hindi* but almost certainly not *Hinglish code-mixed* speech well. **Nobody has published a Hinglish turn detector. That gap is your submission.**

---

## Part 3 — Anatomy of the Reference Implementation (Smart Turn v3.2)

Pipecat's Smart Turn is your starting point and strongest baseline. **Confirmed: the "sample dataset" link in the Shiprocket brief resolves to `pipecat-ai/smart-turn-data-v3.2-train` — the exact data Smart Turn v3 was trained on.**

### 3.1 Input contract
- 16 kHz mono PCM, float32 in [-1, 1]
- Up to **8 seconds** per segment; shorter segments **zero-padded at the front** (audio sits at the *end* of the vector); longer ones truncated to the **last 8s** — keep the ending, drop the beginning
- The single function `truncate_audio_to_last_n_seconds()` is the load-bearing contract shared by training, calibration, and inference

### 3.2 Model

```
log-mel spectrogram (batch, 80 mel, 800 frames ≈ 8s @ 16kHz)
  → Whisper Tiny ENCODER only (384-dim)
  → last_hidden_state (batch, ~1500 tokens, 384)
  → Attention Pooling: Linear(384→256) → Tanh → Linear(256→1) → softmax over tokens
      → weighted-sum pooled vector (batch, 384)
  → Classifier: Linear(384→256) → LayerNorm → GELU → Dropout(0.1)
                → Linear(256→64) → GELU → Linear(64→1)
  → sigmoid → P(complete)   (1 = end of turn = agent should respond)
```

- **Attention pooling** (not mean pooling) lets the model learn to focus on prosody-bearing regions — the trailing intonation and fillers.
- **Loss:** BCE with `pos_weight` from the actual batch class ratio (clamped 0.1–10).
- **Training recipe:** lr 5e-5, 4 epochs, cosine schedule, warmup 0.2, weight decay 0.01.

### 3.3 Deployment
- FP32 ONNX (~32MB) for GPU; **int8 static-quantized ONNX (~8MB)** for CPU
- Quantization: QDQ format, QUInt8 activations / QInt8 weights, per-channel, Entropy calibration, 1024-sample calibration set — **drawn from the training split (a methodological flaw you can fix)**

### 3.4 Inference pattern (production demo)
```
pyaudio 512-sample chunks → Silero VAD (speech prob > 0.5 trigger)
  → ring buffer keeps 200ms pre-speech
  → accumulate segment until 1000ms trailing silence (or cap)
  → run Smart Turn ONCE on the full segment
  → verdict + probability + inference time
```
The model **only runs during silence**, always on the full turn (not incremental chunks).

### 3.5 The dataset — audited with real numbers

Stats from the HF datasets-server (statistics computed on a 35,915-row partial sample; full-set numbers extrapolated):

| Fact | Value |
|---|---|
| Train split | **270,946 rows, 41.4 GB** (~153 KB/sample) |
| Test split | 31,527 rows, 4.84 GB |
| Clip duration | mean 7.6s, median 7.1s, range 0.36–32.6s (most mass under 10s) |
| Label balance | ~50:50 (18,008 incomplete / 17,907 complete in sample) |
| Synthetic share | **~83% TTS-generated**; only ~1.6% human recordings (human_5: 469, human_convcollector_1: 96 in sample) |
| Dominant source | `chirp3_1` (Google Chirp TTS) ≈ 54% of samples; plus chirp3_2, liva, midcentury, rime, orpheus, mundo |
| Languages | 23; sample shares: eng 24%, spa 6%, hin 4.5%, ben ~3%, mar ~2.5%, ~1k each for 18 others |
| Extrapolated full-set counts | **hin ≈ 12.3k samples (~1.9 GB)**, eng ≈ 65k (~10 GB), ben ≈ 7.9k, mar ≈ 6.7k |
| Sub-labels | `midfiller` (True ≈ 41%), `endfiller` (True ≈ 26%) — per-category eval slices built in |

**Three conclusions that drive the whole strategy:**
1. **You cannot download this dataset.** At 41.4GB it is 20× your cap. Even "all the Hindi" (~1.9GB) barely fits. Subsetting is mandatory — see Part 5.
2. **It is overwhelmingly clean studio TTS.** Real Hinglish phone audio (code-mixed, noisy, 8kHz telephony) is essentially absent. A model trained only on this will look good on its own test split and degrade in production. This gap is your thesis.
3. **Smart Turn v3 will be near-ceiling on the official test split** (it trained on this distribution). Your wins must come from (a) Hinglish/realistic eval sets where it degrades, and (b) architecture/training upgrades on subsets.

### 3.6 Known weaknesses (from the codebase analysis — your opportunities)
1. Calibration set drawn from training split, not held out
2. Attention pooling captures weighted **mean only** — discards variance
3. Whisper Tiny is a 2022 English-centric ASR encoder — ASR training strips prosody, and it was not trained on Indian speech
4. Silero VAD lags several hundred ms on speech→non-speech transitions
5. Fixed 8s window, full recompute every trigger
6. ~83% synthetic training data — real-speech robustness unproven
7. No text/context conditioning (sketched in their README, unimplemented)

---

## Part 4 — The Upgrade Map (Component by Component)

Each component: current choice → better alternative → effort → expected gain.

### 4.1 Encoder backbone — *the biggest lever, and the Hinglish experiment*

| Candidate | Size | Verdict |
|---|---|---|
| **Keep Whisper Tiny** (baseline) | ~8M enc | Your control arm. Required. |
| **AI4Bharat IndicConformer 30M (real-time)** | 30M | Trained on actual Indian speech (KathBath/Shrutilipi/MUCS), built for real-time on-device use, MIT license. **Strongest Hinglish-relevant backbone candidate.** 600M multilingual and 120M per-language versions also exist (teacher material). |
| **IndicWav2Vec** (AI4Bharat SSL) | ~95M | SSL representations preserve prosody/paralinguistics better than ASR encoders — and turn detection is fundamentally a prosody task. Worth one ablation arm **if time permits**. |
| **Whisper Small encoder** | ~88M enc | Not a deployment candidate — use as the **distillation teacher** (see 4.5). |
| SenseVoice-Small | 234M | Tempting (encoder-only, 70ms/10s, built-in emotion+event detection) — **but no Hindi support**, non-OSI license. ❌ |
| Moonshine tiny/base/v2 | 26–245M | MIT, 5× less compute than whisper-tiny — **but no Indic language support**. ❌ |
| Parakeet/Canary | 600M+ | European languages only, too big. ❌ |

**Feasibility on RTX 4060 8GB:** Whisper Tiny fine-tune fits comfortably (batch 32–64, fp16). Whisper Small teacher fits (batch 16–32). IndicConformer-30M fits. IndicWav2Vec-95M fits with small batches + gradient accumulation. All viable.

### 4.2 Pooling: attention pooling → **Attentive Statistics Pooling (ASP)**
- **Current:** single-query attention → weighted mean only.
- **Upgrade:** also compute the **weighted standard deviation** from the same attention weights; concat [mean ‖ std] (384→768) into the classifier.
- **Why:** ASP is the standard in speaker verification and speech emotion recognition, consistently beating attention-mean pooling. Turn completeness lives in the *variation* of the final intonation contour — exactly what a mean throws away.
- **Effort: ~10 lines. Cheapest upgrade in the stack.** Code in Part 7.

### 4.3 Classifier head
- **End-biased signal:** concat the pooled vector with a mean of the last ~50 encoder frames (turn completeness is decided by the final 1–2s: trailing intonation, filler, final-word type). ~15 lines.
- **3-class output** (complete / incomplete-pause / explicit-hold like "ek second ruko") — requires data; mark as stretch/future work.
- LSTM/GRU head: Smart Turn tried LSTM already (per their README); skip unless time remains.

### 4.4 Loss function — mostly leave it alone
- Dataset is ~50:50, so `pos_weight` is nearly inert and **focal loss is the wrong tool** here.
- **Do add:** label smoothing 0.05–0.1 — turn-detection labels are inherently noisy at boundaries; smoothing regularizes overconfident boundary errors. Two lines.
- The real lever is **per-language/domain threshold calibration at inference**, not the loss.

### 4.5 Knowledge distillation — *high-ROI training change*
- Fine-tune a bigger encoder (Whisper Small, or IndicConformer) on the same task as **teacher**; train the Tiny student on **soft teacher logits + hard labels** (temperature 2–5, α≈0.5).
- Standard results: lose only 1–2% absolute accuracy vs the teacher at a fraction of the footprint; soft targets also smooth label noise.
- Industry-current: LiveKit distilled Qwen2.5-7B → 0.5B for their multilingual model.
- **Effort: moderate** — one extra teacher run + custom loss. Fits the 4060.

### 4.6 VAD gate: Silero → **TEN VAD**
- TEN VAD (2025, open source): beats Silero on precision-recall on a manually-annotated testset, lower RTF, 306KB. Agent-relevant difference: **Silero lags several hundred ms on speech→non-speech transitions** — exactly the moment your turn detector fires.
- Same 16kHz frame-level interface — **drop-in replacement**. Hours of work.

### 4.7 Quantization & deployment
1. **Held-out calibration set** (dedicated dev split) — one-line data change, fixes a real methodological flaw in the reference.
2. **QAT** to claw back the ~1% int8 accuracy drop — optional.
3. **AVX2 saturation edge case:** ONNX Runtime's U8S8 path can saturate on AVX2/AVX512 — if you see an accuracy cliff on x86, use `reduce_range` or U8U8. Great "we actually deployed it" detail.
4. Try **dynamic quantization** as a 5-minute baseline to quantify what static buys you.

### 4.8 Data pipeline — see Part 5 (now a full strategy, sized to your 2GB cap)

### 4.9 Priority summary

| Priority | Change | Effort | Expected gain |
|---|---|---|---|
| 1 | Hinglish data construction (Part 5) | 1–2 days | **The entire point of the challenge** |
| 2 | TEN VAD replaces Silero | hours | Latency + trigger timing |
| 3 | Attentive Statistics Pooling | hours | Prosodic variance capture |
| 4 | Held-out calibration set | minutes | Valid int8 evaluation |
| 5 | Distillation from Whisper Small teacher | 1 day | Accuracy at same footprint |
| 6 | IndicConformer backbone ablation | 1–2 days | The Hinglish-differentiated experiment (if time) |
| 7 | Label smoothing + telephony augmentation | hours | Robustness |
| Skip | Focal loss; SenseVoice/Moonshine backbones; streaming encoder rework | — | Wrong tool / no Hindi / out of scope |

---

## Part 5 — Data Strategy (the 2GB budget)

### 5.1 The math

- Smart Turn data averages **~153 KB/sample** (FLAC, mean 7.6s). So:
  - **1 GB ≈ ~6,500 samples · 2 GB ≈ ~13,000 samples**
- 13k labeled samples is enough to fine-tune a frozen-encoder classification head well, but not to train from scratch — which is fine, the plan is fine-tuning.
- **Augmentation is applied on-the-fly during training** (noise, speed, telephony bandpass) — it multiplies effective data at zero disk cost. This is how a 2GB cap stops being the bottleneck.

### 5.2 Budget allocation (~1.9 GB)

| # | Source | Samples | Est. size | Role |
|---|---|---|---|---|
| 1 | v3.2-train, `hin` only | 6,000 | ~0.92 GB | Base supervised signal, ready-made labels |
| 2 | v3.2-train, `eng` | 2,000 | ~0.31 GB | Whisper-Tiny's strong language; code-switch anchor |
| 3 | v3.2-train, `mar`+`ben` | 800 | ~0.12 GB | Indic prosody diversity (cheap regularizer) |
| 4 | **Self-built TTS Hinglish** | 3,000–4,000 clips × ~4–6s | ~0.25 GB | **The actual target domain — code-mixed speech** |
| 5 | v3.2-test subset (`hin`+`eng`) | ~1,200 | ~0.18 GB | In-distribution eval (comparable to Pipecat's numbers) |
| 6 | MUCS 2021 Hinglish subset | ~1–2 hours | ~0.10 GB | **Real code-switched human speech — held-out realism eval** |
| | **Total** | | **~1.9 GB** | |

Levers if you need to shrink: cut `hin` to 4,500 (−0.23GB), cut `eng` to 1,000 (−0.15GB), halve the TTS set.

**Subsampling rules** (preserve what makes the dataset good):
- Keep the **50:50 endpoint balance within each language** — sample complete/incomplete equally.
- Preserve **endfiller/midfiller diversity** — don't accidentally take only `nofiller` samples.
- Prefer shorter clips when choosing between equals: mean duration 7.6s but the model only ever sees the last 8s, and shorter clips = more samples per GB.

### 5.3 Acquisition: subsetting a 41GB dataset without downloading it

Option A — **streaming filter** (simple; disk-friendly, but network traffic is larger than the kept data since filtered row-groups still stream through):

```python
from datasets import load_dataset

targets = {"hin": 6000, "eng": 2000, "mar": 400, "ben": 400}
counts = {k: 0 for k in targets}

ds = load_dataset("pipecat-ai/smart-turn-data-v3.2-train",
                  split="train", streaming=True)
kept = []
for ex in ds:
    lang = ex["language"]
    if lang in targets and counts[lang] < targets[lang]:
        kept.append(ex); counts[lang] += 1
    if all(counts[k] >= targets[k] for k in targets):
        break
# then balance endpoint_bool within each language bucket before saving
```

Option B — **server-side filter + per-row audio fetch** (traffic ≈ disk usage; preferred if your cap is really about bandwidth):
1. Query the datasets-server `/filter` endpoint with `where=language='hin'` (and `endpoint_bool`) to get row IDs.
2. Fetch audio for selected rows via the `/rows` endpoint (audio comes as individual URLs) until you hit your per-language quota.

Do the same against `pipecat-ai/smart-turn-data-v3.2-test` for eval item #5.

### 5.4 Building the Hinglish set (the part that makes the submission yours)

**Source 1 — TTS-synthesized Hinglish (edge-tts, free).** This is exactly how Pipecat built 83% of their data — you are replicating their method for a language pair they skipped.

- **Voices:** `hi-IN-SwaraNeural`, `hi-IN-MadhurNeural`, `en-IN-NeerjaNeural`, `en-IN-PrabhatNeural` (edge-tts). Write code-mixed text; Indian-English voices handle embedded Hindi words passably, and vice versa.
- **Script domain: logistics/customer care** — Shiprocket's actual use case. ~600–800 template utterances × voice/speed variations:
  - *Dictation pauses:* "mera phone number hai nine eight seven… [pause] …three two one" → **incomplete at the pause**
  - *Complete answers:* "kal deliver kar dena" / "yes, cancel the order" → **complete**
  - *Contemplative pauses:* "umm… actually pata nahi, thoda sochna padega" → **incomplete at "umm…"**
  - *Hold phrases:* "ek second ruko" / "hold on, let me check" → **incomplete** (and note the future 3-class extension)
  - *Connective endings:* "…aur", "…but", "…toh", "…matlab" → **incomplete**
- **Complete/incomplete pairs:** for each script, synthesize the full utterance (label: complete) and a clause-boundary-truncated version, optionally with an appended filler (label: incomplete). This mirrors Pipecat's conventions exactly.
- **Annotation conventions (steal these from their contribution guide):** end each clip with ~200ms silence; never cut mid-word; keep 50:50 balance; record `midfiller`/`endfiller` sub-labels so your per-category eval works.
- Split speakers/scripts across train/dev/test — **no script overlap between train and test**, or your eval is meaningless.

**Source 2 — MUCS 2021 (real code-switched Hindi-English speech).** Microsoft's code-switching corpus — real human Hinglish conversations with transcripts. You don't train on it; you **evaluate** on it:
- Take ~1–2 hours, segment into utterance-level clips.
- *Complete* examples: clips ending at annotated utterance/turn boundaries.
- *Incomplete* examples: cut continuous speech at a **word boundary** (use corpus word timestamps or forced alignment) mid-utterance — never mid-word.
- This is your "does it survive the real world?" eval set. Expect Smart Turn zero-shot to degrade here vs the official test split — that gap, quantified, is a headline result.

**Source 3 (optional) — KathBath (AI4Bharat).** Large read-speech Hindi corpus. Only if you want extra human Hindi diversity for eval; not required.

### 5.5 On-the-fly augmentation (train-time, zero disk)

- **Telephony simulation:** downsample to 8kHz → bandpass 300–3400Hz → upsample to 16kHz. Non-negotiable for this domain.
- **Noise:** add background noise/babble at 10–20dB SNR (a small free noise pack, ~50MB).
- **Speed perturbation:** 0.9× / 1.1×.
- Apply with probability ~0.5 per sample, per epoch, in the Dataset `__getitem__` — same pattern as Smart Turn's on-demand feature extraction.

### 5.6 Final dataset card (what your report's data section should contain)

| Split | Composition | Size | Purpose |
|---|---|---|---|
| train | hin 6k + eng 2k + mar/ben 0.8k (Smart Turn) + TTS Hinglish ~3k | ~11–12k samples | training |
| dev | 10% held out from the above, stratified by language × label | ~1.2k | early stop, **quantization calibration (held-out!)** |
| test-A (in-distribution) | v3.2-test hin+eng subset | ~1.2k | comparability with Pipecat |
| test-B (domain) | held-out TTS Hinglish scripts | ~600 | target domain |
| test-C (realism) | MUCS subset | ~1–2 hrs | real human code-switched speech |

Three test sets, each answering a different question — this structure alone puts you ahead of most submissions.

---

## Part 6 — The Roadmap (0 → Hero, 1 week, RTX 4060)

### Day 1 — Audit & data acquisition
- [ ] Verify sizes at download time (stats above come from a partial sample — confirm per-language counts as you stream)
- [ ] Pull the subsets per Part 5.2/5.3; verify per-language 50:50 balance
- [ ] **Listen to 50+ samples** (hin especially: pure Hindi or code-mixed? how natural are TTS fillers?). Keep a listening log — it becomes a report section
- [ ] Download Smart Turn v3 int8 ONNX + repo; get `predict.py` running

### Day 2 — Baselines + TTS pipeline
- [ ] **Silence-threshold baseline** on all three test sets (sweep T)
- [ ] **Smart Turn v3 zero-shot** on test-A/B/C — expect strong A, weaker B, weakest C. This table is your thesis
- [ ] Write the Hinglish script generator + edge-tts synthesis pipeline; generate a small pilot batch (100 clips), listen, iterate on prompt/voice quality
- [ ] Evening: launch full TTS generation (~3–4k clips)

### Day 3 — Reproduce & first training run
- [ ] Implement the Smart Turn recipe: Whisper Tiny encoder (frozen) → attention pooling → MLP head, BCE
- [ ] Sanity: overfit 100 samples → loss → 0; then train on the full train split
- [ ] On-the-fly augmentation wired in (telephony/noise/speed)
- [ ] First numbers on test-A/B/C vs the Smart Turn zero-shot row

### Day 4 — Upgrades, one ablation arm each
- [ ] ASP pooling (mean+std)
- [ ] End-biased pooling (concat last-50-frame mean)
- [ ] Label smoothing 0.05–0.1
- [ ] Unfreeze last 1–2 encoder blocks if head-only plateaus
- [ ] **Ablation table started** — one change per row, all three test sets

### Day 5 — Distillation + deployment
- [ ] Whisper Small teacher run (same data), then KD into the Tiny student
- [ ] ONNX export + int8 static quant with **held-out calibration set**; report FP32 vs int8 delta
- [ ] Swap Silero → TEN VAD in the live demo script; tighten trailing-silence cutoff
- [ ] Stretch (only if ahead): IndicConformer-30M backbone ablation

### Day 6 — Evaluation & error analysis
- [ ] **Latency-vs-accuracy curve:** performance with 0.5/1/2/4/8s trailing context
- [ ] **Policy sweep:** threshold → (false-interruption rate, added latency) Pareto frontier, eot-bench style
- [ ] Per-category metrics: by filler type, language, code-switch density, test set
- [ ] **Error analysis:** pull 20+ failures from test-B/C, categorize, write up

### Day 7 — Demo & report
- [ ] Gradio demo on HF Spaces: mic/file → verdict + probability + inference ms; bonus: simulated streaming
- [ ] Weights → HF Hub; README with results table + reproduce commands
- [ ] Final report (write it yourself): framing → data audit → approach → ablations → error analysis → limitations → next steps

**Timebox rule:** if Day 5 ends without distillation working, ship the best ablation-arm model. A clean reproduction + ASP + Hinglish data + rigorous eval beats a half-finished KD experiment.

---

## Part 7 — Key Code Skeletons

### 7.1 Attentive Statistics Pooling (drop-in upgrade)

```python
class AttentiveStatsPooling(nn.Module):
    """Weighted mean + weighted std over encoder frames."""
    def __init__(self, dim=384, attn_dim=256):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, attn_dim), nn.Tanh(), nn.Linear(attn_dim, 1)
        )

    def forward(self, hidden, mask=None):
        # hidden: (B, T, D); mask: (B, T) with 1 = valid frame
        w = self.attn(hidden)                        # (B, T, 1)
        if mask is not None:
            w = w.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        w = torch.softmax(w, dim=1)                  # attention weights
        mean = (w * hidden).sum(dim=1)               # weighted mean (B, D)
        var = (w * (hidden - mean.unsqueeze(1)) ** 2).sum(dim=1)
        std = torch.sqrt(var.clamp(min=1e-5))
        return torch.cat([mean, std], dim=1)         # (B, 2D) -> classifier input 768
```

Classifier then starts at `Linear(768, 256)`. Everything else unchanged.

### 7.2 End-biased concatenation

```python
last_k = hidden[:, -50:, :].mean(dim=1)            # last ~1s of encoder frames
pooled = pool(hidden)                              # ASP output (B, 768)
features = torch.cat([pooled, last_k], dim=1)      # -> Linear(1152, 256)
```

### 7.3 Distillation loss

```python
def kd_loss(student_logits, teacher_logits, labels, T=3.0, alpha=0.5):
    soft = F.binary_cross_entropy_with_logits(
        student_logits / T, torch.sigmoid(teacher_logits / T)) * (T * T)
    hard = F.binary_cross_entropy_with_logits(student_logits, labels)
    return alpha * soft + (1 - alpha) * hard
```

Teacher forward under `torch.no_grad()`; precompute teacher logits once per dataset to save compute.

### 7.4 Telephony augmentation (train-time)

```python
def telephony(waveform, sr=16000, p=0.5):
    if random.random() > p:
        return waveform
    x = torchaudio.functional.resample(waveform, sr, 8000)
    x = torchaudio.functional.bandpass_biquad(x, 8000, 300, 3400)
    return torchaudio.functional.resample(x, 8000, sr)
```

### 7.5 Latency-vs-accuracy evaluation harness

```python
for context_sec in [0.5, 1.0, 2.0, 4.0, 8.0]:
    probs = [predict(truncate_to_last_n_seconds(a, context_sec)) for a in test_audio]
    metrics[context_sec] = compute_metrics(probs, labels)
# plot: x = context seconds (proxy for decision latency), y = F1
```

### 7.6 Policy sweep (eot-bench style)

```python
for threshold in np.arange(0.5, 0.99, 0.05):
    preds = (probs >= threshold).astype(int)
    false_interrupt = ((preds == 1) & (labels == 0)).mean()   # cut user off
    missed_eot      = ((preds == 0) & (labels == 1)).mean()   # dead air
# plot Pareto frontier: false-interrupt rate vs implied added latency
```

---

## Part 8 — Deliverables Checklist

**GitHub repo**
- [ ] README: problem, approach, results table, architecture diagram, how to run
- [ ] `train.py` / `inference.py` / `evaluate.py` / `data/` (subset + TTS generation scripts), pinned `requirements.txt`, seeds
- [ ] `results.md` with exact reproduce commands
- [ ] Model weights (also on HF Hub)

**Demo (HF Spaces / Gradio)**
- [ ] Mic + file upload → verdict, probability, inference ms
- [ ] Simulated streaming playback (bonus, high impact)

**Report (write it yourself — they explicitly asked)**
- [ ] Problem framing: why turn detection is hard; why audio-native for Hinglish
- [ ] Dataset audit: the 83%-TTS / 1.6%-human finding, your listening log, the Hinglish gap
- [ ] Data strategy: budget math, subset rules, TTS generation method, three-test-set design
- [ ] Approach: architecture + why each choice
- [ ] Experiments: ablation table, latency-accuracy curves, policy frontier, FP32 vs int8
- [ ] Error analysis: categorized failures with audio examples
- [ ] Limitations + next steps: streaming encoder, 3-class hold phrases, text conditioning, QAT, scaling the human Hinglish data

---

## Part 9 — Strategic Positioning (read before writing the report)

1. **Your headline table is Smart Turn v3 zero-shot vs your model across test-A/B/C.** Smart Turn should win test-A (its own training distribution); your model should win B and C. That crossover *is* the story: in-distribution benchmarks overstate production readiness.
2. **The dataset audit is a finding, not a chore.** "The reference dataset is 83% synthetic TTS, 1.6% human, and its Hindi is not code-mixed Hinglish" — quantifying this shows you did the data-prep depth they asked for.
3. **Frame "why audio-native" with evidence:** code-switching breaks ASR (14–16% WER on telephony audio), ASR strips fillers, text discards prosody → audio-native is the architecturally correct choice for this market.
4. **The TTS-Hinglish pipeline mirrors Pipecat's own method** — you're not cutting corners, you're applying their proven data-generation approach to the language pair they skipped. Say so.
5. **Show policy-layer maturity:** probabilities → thresholds → Pareto frontier. Most candidates report one accuracy number; the field evaluates policies.
6. **Steal Smart Turn's annotation conventions** verbatim for your TTS data: prosody-equal-to-words, never-cut-mid-word, 200ms trailing silence, filler sub-labels.
7. **Timebox ruthlessly.** A clean Whisper-Tiny reproduction + ASP + Hinglish data + rigorous evaluation beats a half-finished IndicConformer experiment.

---

## Part 10 — Resources

| Resource | Where |
|---|---|
| Smart Turn v3 (code, weights) | `github.com/pipecat-ai/smart-turn` · `huggingface.co/pipecat-ai/smart-turn-v3` |
| Smart Turn data (**subset per Part 5**) | `huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train` (+ `-test`) |
| TEN VAD | `github.com/ten-framework/ten-vad` |
| Silero VAD | `github.com/snakers4/silero-vad` |
| edge-tts (Hinglish synthesis) | `github.com/rany2/edge-tts` — voices: hi-IN-Swara/Madhur, en-IN-Neerja/Prabhat |
| MUCS 2021 (real code-switched Hinglish) | Microsoft Speech Corpus — `microsoft.com/en-us/research` (MUCS 2021 challenge data) |
| KathBath / Indic corpora | AI4Bharat — `github.com/AI4Bharat` · ai4bharat.iitm.ac.in |
| AI4Bharat models (IndicConformer, IndicWav2Vec) | `github.com/AI4Bharat/Indic-Conformer-ASR` |
| LiveKit turn detector + eot-bench | `github.com/livekit` · blog at livekit.io |
| VAP (voice activity projection) | `github.com/ErikEkstedt/VAP` |
| TurnGPT | `github.com/ErikEkstedt/TurnGPT` |

---

*Playbook ends. Rule of thumb: when in doubt, prefer the option that generates a table or a curve for the report.*
