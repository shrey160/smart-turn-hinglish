"""Generate TTS Hinglish clips (edge-tts) — logistics/customer-care domain.

Design (masterplan P2 / playbook Day 2):
- 25 handwritten scripts, each with two renderings: hi (Devanagari + English
  tokens) for hi-IN voices, en (Roman Hinglish) for en-IN voices.
- Complete/incomplete PAIRS from shared templates via markers:
    `|`  clause cut point (drop everything after) -> incomplete
    `^`  midfiller insertion point
  Variants: complete (optionally midfiller) | cut | endfiller | midfiller.
- 4 voices rotated per script: hi-IN-SwaraNeural, hi-IN-MadhurNeural,
  en-IN-NeerjaNeural, en-IN-PrabhatNeural.
- Output: FLAC 16 kHz mono, speech + 250 ms tail silence, <= 8 s total,
  Smart Turn folder layout data/tts_hinglish/<split>/<label-folder>/.
- Full manifest data/tts_hinglish/manifest.csv (resumable: existing rows skipped).

Pilot: 25 scripts x (complete, incomplete) x 2 fills = 100 clips, 50:50 labels.

Run:  uv run python scripts/generate_tts_hinglish.py --pilot
"""
import argparse
import asyncio
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "tts_hinglish"
MANIFEST = OUT / "manifest.csv"
TAIL_S = 0.25
MAX_S = 8.0
SR = 16000

VOICES = [
    ("hi-IN-SwaraNeural", "swara", "hi"),
    ("hi-IN-MadhurNeural", "madhur", "hi"),
    ("en-IN-NeerjaNeural", "neerja", "en"),
    ("en-IN-PrabhatNeural", "prabhat", "en"),
]
FILL_HI = ["उम्...", "हम्म...", "आ..."]
FILL_EN = ["umm...", "hmm...", "aa..."]

# id, domain, hi rendering, en rendering. `|` = final-clause cut point,
# `^` = midfiller insertion point (mid-utterance, BEFORE the cut marker).
SCRIPTS = [
    ("s01", "track", "नमस्ते, ^ मेरा order SR{oid} दो दिन से out for delivery दिखा रहा है|, कल तक पहुँच जाएगा?",
     "Hello, ^ mera order SR{oid} do din se out for delivery dikha raha hai|, kal tak pahunch jayega?"),
    ("s02", "delay", "देखिए, ^ मेरा parcel {n} दिन late हो गया है|, delivery date बार बार change हो रही है.",
     "Dekhiye, ^ mera parcel {n} din late ho gaya hai|, delivery date baar baar change ho rahi hai."),
    ("s03", "return", "मैंने SR{oid} के लिए return request डाली थी, ^ pickup कब होगा| बता दीजिए.",
     "Maine SR{oid} ke liye return request daali thi, ^ pickup kab hoga| bata dijiye."),
    ("s04", "refund", "Return पिक हो गया था, ^ पर refund अभी तक नहीं आया|, पैसे कब तक credit होंगे?",
     "Return pick ho gaya tha, ^ par refund abhi tak nahi aaya|, paise kab tak credit honge?"),
    ("s05", "address", "Shipping address गलत डाल दिया मैंने, ^ pin code भी wrong है|, अभी update कर सकते हैं?",
     "Shipping address galat daal diya maine, ^ pin code bhi wrong hai|, abhi update kar sakte hain?"),
    ("s06", "cancel", "Order SR{oid} cancel करना चाहती हूँ, ^ वो अभी shipped नहीं हुआ है ना|?",
     "Order SR{oid} cancel karna chahti hoon, ^ wo abhi shipped nahi hua hai na|?"),
    ("s07", "exchange", "Kurta का size बड़ा है, ^ exchange करना है मुझे|, process क्या होगा?",
     "Kurta ka size bada hai, ^ exchange karna hai mujhe|, process kya hoga?"),
    ("s08", "cod", "COD order था मेरा, ^ अब online payment करना चाहता हूँ|, option कहाँ मिलेगा?",
     "COD order tha mera, ^ ab online payment karna chahta hoon|, option kahan milega?"),
    ("s09", "damaged", "Box खोला तो अंदर का item टूटा हुआ था, ^ photo भी भेज दी है आपको|, refund मिलेगा?",
     "Box khola to andar ka item toota hua tha, ^ photo bhi bhej di hai aapko|, refund milega?"),
    ("s10", "wrong", "मेरे order में गलत product आया है, ^ मैंने black colour माँगा था|, blue भेज दिया.",
     "Mere order mein galat product aaya hai, ^ maine black colour maanga tha|, blue bhej diya."),
    ("s11", "invoice", "Invoice में GST number गलत आया है, ^ corrected invoice चाहिए| कल तक.",
     "Invoice mein GST number galat aaya hai, ^ corrected invoice chahiye| kal tak."),
    ("s12", "reschedule", "कल मैं घर पर नहीं रहूँगा, ^ delivery परसों कर दीजिए|, evening slot में.",
     "Kal main ghar par nahin rahunga, ^ delivery parso kar dijiye|, evening slot mein."),
    ("s13", "ndr", "Delivery boy ने call किया था, ^ मैं meeting में थी तो नहीं उठा पाई|, अब क्या करूँ?",
     "Delivery boy ne call kiya tha, ^ main meeting mein thi to nahin utha payi|, ab kya karoon?"),
    ("s14", "otp", "OTP आ ही नहीं रहा मेरे नंबर पर, ^ तीन बार resend किया है मैंने|, क्या problem हो सकती है?",
     "OTP aa hi nahi raha mere number par, ^ teen baar resend kiya hai maine|, kya problem ho sakti hai?"),
    ("s15", "app", "आपकी app में tracking page load नहीं हो रहा, ^ हर बार error आता है|, order SR{oid} का.",
     "Aapki app mein tracking page load nahi ho raha, ^ har baar error aata hai|, order SR{oid} ka."),
    ("s16", "weight", "Courier ने weight dispute डाला है, ^ actual weight 800 gram है|, 2 kg कैसे हुआ?",
     "Courier ne weight dispute daala hai, ^ actual weight 800 gram hai|, 2 kg kaise hua?"),
    ("s17", "charges", "Freight charges extra कटे हैं, ^ agreement के हिसाब से ये नहीं बनता|, refund करवाइए.",
     "Freight charges extra kate hain, ^ agreement ke hisaab se ye nahin banta|, refund karwaiye."),
    ("s18", "lost", "पिछले हफ्ते pickup हुआ था, ^ तब से tracking update नहीं आया|, कहीं lost तो नहीं?",
     "Pichhle hafte pickup hua tha, ^ tab se tracking update nahin aaya|, kahin lost to nahin?"),
    ("s19", "partial", "दो boxes थे मेरे order में, ^ एक ही पहुँचा है अभी तक|, दूसरा कब आएगा?",
     "Do boxes the mere order mein, ^ ek hi pahuncha hai abhi tak|, doosra kab aayega?"),
    ("s20", "open_box", "Delivery से पहले open box delivery चाहिए, ^ checking करनी है item की|, possible है?",
     "Delivery se pehle open box delivery chahiye, ^ checking karni hai item ki|, possible hai?"),
    ("s21", "seller", "मैं seller हूँ, ^ मेरा shipment pickup pending दिखा रहा है|, तीन दिन से.",
     "Main seller hoon, ^ mera shipment pickup pending dikha raha hai|, teen din se."),
    ("s22", "warehouse", "Warehouse से आपके लोगों ने call किया था, ^ manifest की problem बता रहे थे|, मैं समझ नहीं पाया.",
     "Warehouse se aapke logon ne call kiya tha, ^ manifest ki problem bata rahe the|, main samajh nahin paya."),
    ("s23", "nri", "मुझे international shipping चाहिए, ^ Dubai में भेजना है एक parcel|, rate क्या लगेगा?",
     "Mujhe international shipping chahiye, ^ Dubai mein bhejna hai ek parcel|, rate kya lagega?"),
    ("s24", "complaint", "चार बार complaint कर चुका हूँ, ^ ticket बंद कर देते हैं बिना solution के|, senior से बात कराइए.",
     "Chaar baar complaint kar chuka hoon, ^ ticket band kar dete hain bina solution ke|, senior se baat karaiye."),
    ("s25", "thanks", "Order समय पर पहुँच गया, ^ packaging अच्छी थी|, thanks बोलना था बस.",
     "Order time par pahunch gaya, ^ packaging acchi thi|, thanks bolna tha bas."),
]
KINDS = ["cut", "endfiller", "midfiller"]


def render2(template: str, fill: dict, kind: str, fillers: list, rng: random.Random) -> str:
    """Marker transform, then slot fill. kind in {complete, cut, endfiller, midfiller}."""
    filler = rng.choice(fillers)
    if kind == "complete":
        text = template.replace("^", "").replace("|", ",")
        text = re.sub(r",\s*,", ",", text)  # collapse "|," -> ",," doubles
        text = re.sub(r",\s*([?.!])", r"\1", text)  # "ना,?" -> "ना?"
    elif kind in ("cut", "endfiller", "midfiller"):
        text = template.split("|")[0].rstrip(" ,।,.|")
        if kind == "endfiller":
            text = text + " " + filler
        if kind == "midfiller":
            text = text.replace("^", filler + " ")
    text = text.replace("^", "").format(**fill)
    return " ".join(text.split())


LABEL_FOLDERS = {
    ("complete", False): "complete-False-False",
    ("complete", True): "complete-True-False",
    ("cut", None): "incomplete-False-False",
    ("endfiller", None): "incomplete-False-True",
    ("midfiller", None): "incomplete-True-False",
}


async def synth(text: str, voice: str, mp3_path: Path) -> None:
    import edge_tts

    for attempt in range(3):
        try:
            await edge_tts.Communicate(text, voice, rate="+8%").save(str(mp3_path))
            return
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                raise
            print(f"    retry {attempt + 1} after error: {type(e).__name__}")
            await asyncio.sleep(1.5 * (attempt + 1))


def to_flac(mp3_path: Path, flac_path: Path) -> float:
    import librosa

    audio, sr = librosa.load(str(mp3_path), sr=SR, mono=True)
    audio = audio.astype(np.float32)
    audio = np.concatenate([audio, np.zeros(int(TAIL_S * SR), dtype=np.float32)])
    dur = len(audio) / SR
    if dur > MAX_S:
        sf.write(str(flac_path), audio, SR)  # keep + flag in manifest
    else:
        sf.write(str(flac_path), audio, SR)
    return dur


def load_manifest() -> pd.DataFrame:
    if MANIFEST.exists():
        return pd.read_csv(MANIFEST, dtype={"pilot": "boolean"})
    return pd.DataFrame(columns=["file", "split", "pilot", "script_id", "voice", "variant",
                                 "label_folder", "text", "duration_s", "status"])


async def run(jobs, manifest_df) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "_mp3"
    tmp.mkdir(exist_ok=True)
    done = set(manifest_df["file"])
    for i, job in enumerate(jobs):
        if job["file"] in done:
            print(f"  skip existing {job['file']}")
            continue
        label_dir = OUT / job["split"] / job["label_folder"]
        label_dir.mkdir(parents=True, exist_ok=True)
        flac = label_dir / f"{job['file']}.flac"
        mp3 = tmp / f"{job['file']}.mp3"
        print(f"  [{i + 1}/{len(jobs)}] {job['voice']} {job['variant']}: {job['text'][:48]}...")
        try:
            await synth(job["text"], job["voice"], mp3)
            dur = to_flac(mp3, flac)
            status = "ok" if dur <= MAX_S else "too_long"
            mp3.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            print(f"    FAILED: {type(e).__name__}: {e}")
            dur, status = np.nan, "failed"
        manifest_df.loc[len(manifest_df)] = {**job, "duration_s": round(dur, 2) if dur == dur else np.nan,
                                             "status": status}
        await asyncio.sleep(random.uniform(0.3, 0.8))
    manifest_df.to_csv(MANIFEST, index=False)
    return manifest_df


def build_jobs(n_scripts: int, split: str, pilot: bool) -> list:
    rng = random.Random(42)
    jobs = []
    for i, (sid, dom, hi, en) in enumerate(SCRIPTS[:n_scripts]):
        voice_name, voice_short, vlang = VOICES[i % len(VOICES)]
        fillers = FILL_HI if vlang == "hi" else FILL_EN
        template = hi if vlang == "hi" else en
        fills = [{"oid": 2100 + 7 * i, "n": 3 + i % 5}, {"oid": 2100 + 7 * i + 1, "n": 3 + (i + 1) % 5}]
        plan = [
            ("complete", False, fills[0]),
            (KINDS[i % 3], None, fills[0]),
            ("complete", i % 2 == 0, fills[1]),
            (KINDS[(i + 1) % 3], None, fills[1]),
        ]
        for kind, mid, fill in plan:
            text = render2(template, fill, kind, fillers, rng)
            jobs.append({
                "file": f"{sid}_{voice_short}_{kind}{'' if mid is None or not mid else '_mf'}" + f"_o{fill['oid']}",
                "split": split,
                "pilot": pilot,
                "script_id": sid,
                "voice": voice_name,
                "variant": kind if not mid else "complete_mf",
                "label_folder": LABEL_FOLDERS[(kind, mid)] if kind == "complete" else LABEL_FOLDERS[(kind, None)],
                "text": text,
            })
    return jobs


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="100-clip pilot (25 scripts x 4)")
    ap.add_argument("--n-scripts", type=int, default=25)
    ap.add_argument("--split", default="train", choices=["train", "test_b"])
    args = ap.parse_args()
    if not args.pilot:
        raise SystemExit("only --pilot is wired; full-scale generation comes after the listening gate")

    jobs = build_jobs(args.n_scripts, args.split, pilot=True)
    mdf = load_manifest()
    mdf = asyncio.run(run(jobs, mdf))

    print("\n=== pilot stats ===")
    new = mdf[mdf["file"].isin({j["file"] for j in jobs})]
    print(new.groupby(["label_folder", "status"]).size().unstack(fill_value=0).to_string())
    ok = new[new["status"].isin(["ok", "too_long"])]
    print("\nduration:", ok["duration_s"].describe()[["mean", "min", "max"]].round(2).to_dict())
    print("too_long:", (new["status"] == "too_long").sum(), "failed:", (new["status"] == "failed").sum())
    print("sr/channels check via read-back:")
    for f in ok["file"].head(3):
        info = sf.info(str(next((OUT / args.split).rglob(f + ".flac"))))
        print(" ", f, info.samplerate, info.channels)


if __name__ == "__main__":
    main()
