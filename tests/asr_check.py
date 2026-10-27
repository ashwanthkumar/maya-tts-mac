"""Generate a few sentences and transcribe them with Whisper (dev check of intelligibility; run in .venv-ref).

    .venv-ref/bin/python tests/asr_check.py [fp32|bf16|fp16] [seeds...]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402
import torch  # noqa: E402
from transformers import pipeline  # noqa: E402

from maya2_mlx.model import Maya2TTS  # noqa: E402

CASES = [
    ("en", "Siddhant", "english", "The EMI for your home loan is due by 12-10-24; please ensure your account has sufficient funds."),
    ("hi", "Ahana", "hindi", "नमस्ते, आपका ऑर्डर कल पहुँच जाएगा।"),
    ("ta", "Abhay", "tamil", "கடந்த நிதியாண்டில் வங்கியின் மொத்த லாபம் ₹3.5 கோடி, இதில் ₹45 லட்சம் முதலீட்டாளர்களுக்கு வழங்கப்பட்டது."),
]
DT = {"fp32": (mx.float32, mx.float32), "bf16": (mx.bfloat16, mx.float16), "fp16": (mx.float16, mx.float16)}

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "bf16"
    seeds = [int(s) for s in sys.argv[2:]] or [7]
    tts = Maya2TTS.from_pretrained(dtype=DT[mode][0], depth_dtype=DT[mode][1])
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device="cpu", torch_dtype=torch.float32)
    for lang, voice, wl, text in CASES:
        for seed in seeds:
            info = {}
            w = tts.generate(text, lang=lang, voice=voice, seed=seed, info=info)
            r = asr({"raw": w, "sampling_rate": 24000}, generate_kwargs={"language": wl, "task": "transcribe"})["text"].strip()
            print(f"RESULT {mode} {lang} seed={seed} {info['audio_s']:.1f}s rtf={info['rtf']:.2f} "
                  f"stop={info['stops'][-1]['reason']}: {r}", flush=True)
