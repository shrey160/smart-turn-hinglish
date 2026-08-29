"""Build test-C (MUCS real-speech realism eval) from one MUCS-Hinglish test parquet.

Pipeline (SP5, masterplan P3):
  1. Read parquet, profile durations/sr via soundfile (bytes are WAV).
  2. Filter to Hinglish core, 1.5-8.0 s (never triggers the last-8s contract).
  3. complete-False-False -> full utterances, +200 ms trailing silence appended
     (repo convention / v3.2 structure: MUCS trims segments tightly, so without
     the pad completes end mid-speech too and the task has no acoustic cue).
  4. incomplete-False-False -> mid-speech cuts: pick a cut word index in
     [35%, 70%] of the transcript (>=3 words must remain), locate the cut time
     by char-proportional interpolation, then snap to a SPEECH-ACTIVE 20 ms
     frame within +/-0.35 s (clip ends while the user is still talking, like
     v3.2 incompletes — never in a pause), 5 ms microfade.
  5. Balance 50:50 by count, cap per speaker, write WAVs 16 kHz mono + manifest.

Usage:
  uv run python scripts/build_test_c.py            # dry run: selection stats only
  uv run python scripts/build_test_c.py --write    # write data/test_c + manifest

Playbook deviation (logged in HARDPOINT): transcript-guided energy-valley cuts
replace torchaudio forced alignment (MMS_FA ~1.2 GB, no Windows perl for uroman);
remaining-word constraint keeps cuts semantically mid-sentence.
"""

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "data" / "mucs_tmp" / "data" / "test-00000-of-00002.parquet"
OUT = ROOT / "data" / "test_c" / "mucs-hinglish"
MANIFEST = ROOT / "data" / "splits" / "test_c_manifest.csv"
SR = 16000

MIN_DUR, MAX_DUR = 1.5, 8.0
ENG_RATIO_RANGE = (0.02, 0.70)
CUT_FRAC_RANGE = (0.35, 0.70)
MIN_WORDS_REMAIN = 3
MIN_CUT_DUR = 1.2
SPEAKER_CAP = 40
TARGET_PER_CLASS = 550
SEED = 42


def profile_rows():
    pf = pq.ParquetFile(PARQUET)
    rows = []
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(
            rg, columns=["audio", "segment_id", "transcript", "speaker_id", "ratio_english_words"]
        )
        for a, sid, tr, spk, eng in zip(
            t.column("audio").to_pylist(),
            t.column("segment_id").to_pylist(),
            t.column("transcript").to_pylist(),
            t.column("speaker_id").to_pylist(),
            t.column("ratio_english_words").to_pylist(),
        ):
            info = sf.info(io.BytesIO(a["bytes"]))
            rows.append(
                {
                    "bytes": a["bytes"],
                    "segment_id": sid,
                    "transcript": tr,
                    "speaker_id": spk,
                    "eng_ratio": float(eng or 0.0),
                    "sr": info.samplerate,
                    "frames": info.frames,
                    "dur": info.frames / info.samplerate,
                }
            )
    return rows


def decode_16k(row):
    wav, sr = sf.read(io.BytesIO(row["bytes"]), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return wav.astype(np.float32)


def find_speech_onset_frame(wav, t_target, win_s=0.35, frame_ms=20):
    """Frame at/after t_target whose RMS is high (speech ongoing at cut)."""
    frame = int(SR * frame_ms / 1000)
    lo = max(0, int((t_target - win_s) * SR))
    hi = min(len(wav) - frame, int((t_target + win_s) * SR))
    if hi - lo < frame:
        return None
    n = (hi - lo) // frame
    rms = np.sqrt(
        np.mean(wav[lo : lo + n * frame].reshape(n, frame) ** 2, axis=1) + 1e-9
    )
    thresh = np.quantile(rms, 0.6)
    speech = np.where(rms >= thresh)[0]
    if len(speech) == 0:
        return None
    k = int(speech[np.argmin(np.abs(speech - (int(t_target * SR) - lo) // frame))])
    return lo + k * frame


def make_incomplete(wav, transcript, segment_id):
    words = transcript.split()
    if len(words) < MIN_WORDS_REMAIN + 2:
        return None
    import zlib

    rng = np.random.default_rng(zlib.crc32(segment_id.encode()) % 2**32)
    fracs = np.cumsum([len(w) + 1 for w in words]) / (sum(len(w) + 1 for w in words) + 1)
    cfrac = rng.uniform(*CUT_FRAC_RANGE)
    wi = int(np.searchsorted(fracs, cfrac))
    if wi < 2 or len(words) - wi < MIN_WORDS_REMAIN:
        wi = max(2, len(words) - MIN_WORDS_REMAIN)
    t_cut = fracs[wi - 1] * len(wav) / SR
    cut = find_speech_onset_frame(wav, t_cut)
    if cut is None:
        return None
    if cut < MIN_CUT_DUR * SR or len(wav) - cut < 0.3 * SR:
        return None
    out = wav[:cut].copy()
    fade = min(int(0.005 * SR), len(out))
    out[-fade:] *= np.linspace(1.0, 0.85, fade).astype(np.float32)
    return out, wi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    print(f"profile {PARQUET.name} ...")
    rows = profile_rows()
    srs = sorted({r["sr"] for r in rows})
    print(f"rows={len(rows)} sr={srs} total_h={sum(r['dur'] for r in rows)/3600:.2f}")
    print(f"speakers={len({r['speaker_id'] for r in rows})}")

    durs = np.array([r["dur"] for r in rows])
    print(
        "dur pct 5/50/95: %.2f / %.2f / %.2f s"
        % tuple(np.percentile(durs, [5, 50, 95]))
    )

    pool = [
        r
        for r in rows
        if MIN_DUR <= r["dur"] <= MAX_DUR
        and ENG_RATIO_RANGE[0] <= r["eng_ratio"] <= ENG_RATIO_RANGE[1]
    ]
    print(
        f"after filter (dur {MIN_DUR}-{MAX_DUR}s, eng_ratio {ENG_RATIO_RANGE}): {len(pool)} "
        f"({sum(r['dur'] for r in pool)/3600:.2f} h)"
    )

    complete, incomplete = [], []
    for r in pool:
        if rng.random() < 0.5 and len(complete) < 3 * TARGET_PER_CLASS:
            complete.append(r)
            continue
        res = make_incomplete(decode_16k(r), r["transcript"], r["segment_id"])
        if res is not None and len(incomplete) < 3 * TARGET_PER_CLASS:
            incomplete.append((r, res[1]))

    def cap_by_speaker(items, n):
        per, out = {}, []
        order = list(items)
        rng.shuffle(order)
        for it in order:
            spk = it[0]["speaker_id"] if isinstance(it, tuple) else it["speaker_id"]
            if per.get(spk, 0) >= SPEAKER_CAP:
                continue
            per[spk] = per.get(spk, 0) + 1
            out.append(it)
            if len(out) >= n:
                break
        return out

    n = min(TARGET_PER_CLASS, len(complete), len(incomplete))
    complete = cap_by_speaker(complete, n)
    incomplete = cap_by_speaker(incomplete, n)
    assert len({i[0]["segment_id"] if isinstance(i, tuple) else i["segment_id"] for i in complete})
    assert not (
        {i[0]["segment_id"] if isinstance(i, tuple) else i["segment_id"] for i in complete}
        & {i[0]["segment_id"] for i in incomplete}
    ), "class overlap"

    def stats(items, name):
        d = [i[0]["dur"] if isinstance(i, tuple) else i["dur"] for i in items]
        print(
            f"{name}: n={len(items)} dur_sum={sum(d)/60:.1f} min mean={np.mean(d):.2f}s "
            f"speakers={len({(i[0] if isinstance(i, tuple) else i)['speaker_id'] for i in items})}"
        )

    stats(complete, "complete  ")
    stats(incomplete, "incomplete")
    print(f"total: {(sum((i[0] if isinstance(i, tuple) else i)['dur'] for i in complete + incomplete))/60:.1f} min")

    if not args.write:
        print("\nDRY RUN — nothing written. Re-run with --write.")
        return

    out_c = OUT / "complete-False-False"
    out_i = OUT / "incomplete-False-False"
    out_c.mkdir(parents=True, exist_ok=True)
    out_i.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    manifest = []
    for r in complete:
        wav = decode_16k(r)
        wav = np.concatenate([wav, np.zeros(int(0.2 * SR), dtype=np.float32)])
        p = out_c / f"{r['segment_id']}.wav"
        sf.write(p, wav, SR, subtype="PCM_16")
        manifest.append([r["segment_id"], r["speaker_id"], "complete", "", "", r["dur"], r["transcript"], r["eng_ratio"]])
    for r, wi in incomplete:
        out = make_incomplete(decode_16k(r), r["transcript"], r["segment_id"])
        if out is None:
            print(f"skip (cut re-derivation failed): {r['segment_id']}")
            continue
        wav = out[0]
        p = out_i / f"{r['segment_id']}.wav"
        sf.write(p, wav, SR, subtype="PCM_16")
        manifest.append([r["segment_id"], r["speaker_id"], "incomplete", wi, len(wav) / SR, r["dur"], r["transcript"], r["eng_ratio"]])

    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["segment_id", "speaker_id", "label", "cut_word_idx", "clip_dur_s", "orig_dur_s", "transcript", "eng_ratio"])
        w.writerows(manifest)

    n_c = len(list(out_c.glob("*.wav")))
    n_i = len(list(out_i.glob("*.wav")))
    size_mb = sum(p.stat().st_size for p in OUT.rglob("*.wav")) / 1e6
    print(f"\nwrote complete={n_c} incomplete={n_i} size={size_mb:.1f} MB -> {OUT}")
    print(f"manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
