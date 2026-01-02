import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # ✅ TF가 GPU 못 쓰게 (cuDNN 문제 회피)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json, argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub

TARGET_CLASSES = {
    "laughter": ["Laughter"],
    "applause": ["Applause"],
    "cheering": ["Cheering"],
}

def load_wav_16k(path: Path):
    wav, sr = sf.read(str(path))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        raise ValueError("input must be 16k wav")
    return wav.astype(np.float32), sr

def merge_events(events, gap=0.25):
    """같은 reaction_type이 서로 가까우면 병합"""
    if not events:
        return []
    events.sort(key=lambda x: (x["reaction_type"], x["start"], x["end"]))
    merged = []
    cur = dict(events[0])
    for e in events[1:]:
        if e["reaction_type"] == cur["reaction_type"] and e["start"] <= cur["end"] + gap:
            cur["end"] = max(cur["end"], e["end"])
            cur["score"] = max(cur["score"], e.get("score", 0.0))
        else:
            merged.append(cur)
            cur = dict(e)
    merged.append(cur)
    merged.sort(key=lambda x: x["start"])
    return merged

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--chunk_sec", type=float, default=10.0)
    args = ap.parse_args()

    in_wav = Path(args.input)
    out_json = Path(args.output)

    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
    class_map_path = yamnet.class_map_path().numpy().decode("utf-8")
    class_names = [line.strip().split(",")[2] for line in open(class_map_path, "r", encoding="utf-8").read().splitlines()[1:]]

    wav, sr = load_wav_16k(in_wav)
    total_dur = len(wav) / sr

    # YAMNet frame hop은 보통 약 0.48초
    hop = 0.48

    all_events = []
    chunk_samples = int(args.chunk_sec * sr)

    for start_samp in range(0, len(wav), chunk_samples):
        end_samp = min(start_samp + chunk_samples, len(wav))
        chunk = wav[start_samp:end_samp]
        offset_sec = start_samp / sr

        scores, embeddings, spectrogram = yamnet(chunk)
        scores = scores.numpy()
        frames = scores.shape[0]

        for rtype, names in TARGET_CLASSES.items():
            idxs = [i for i, nm in enumerate(class_names) if nm in names]
            if not idxs:
                continue

            rscore = scores[:, idxs].max(axis=1)
            active = rscore >= args.threshold

            i = 0
            while i < frames:
                if not active[i]:
                    i += 1
                    continue
                j = i
                peak = float(rscore[i])
                while j < frames and active[j]:
                    peak = max(peak, float(rscore[j]))
                    j += 1

                s = offset_sec + i * hop
                e = offset_sec + j * hop
                s = min(s, total_dur)
                e = min(e, total_dur)

                if e - s >= 0.3:
                    all_events.append({
                        "start": float(s),
                        "end": float(e),
                        "reaction_type": rtype,
                        "score": float(peak),
                    })
                i = j

    all_events = merge_events(all_events, gap=0.25)

    payload = {
        "video_id": in_wav.stem.replace(".16k", ""),
        "created_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "threshold": args.threshold,
        "events": all_events,
        "note": "CPU-only TF (CUDA_VISIBLE_DEVICES='')"
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
