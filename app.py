"""Stable Audio 3 — Medium, Small Music, Small SFX.
Two tabs:
* **Simple** — prompt + duration with a slim Advanced accordion (steps/CFG/seed
  /sampler). Mirrors the original tiny UI.
* **Advanced** — replicates the reference repo's
  ``stable_audio_3/interface/diffusion_cond.py`` controls: negative prompt,
  sampler params (sigma_max, APG, duration padding), init audio + noise level,
  inpainting with mask start/end, spectrogram gallery, send-to-init /
  send-to-inpaint buttons.
"""

from __future__ import annotations

import spaces  # noqa: F401

import argparse
import json
import os
import logging
import random
import re
import subprocess
import sys
import time
import warnings
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

APP_DIR = Path(__file__).resolve().parent
CKPTS_DIR = APP_DIR / "ckpts"
OUTPUTS_DIR = APP_DIR / "outputs"
SETTINGS_PATH = APP_DIR / "stable_audio_3_settings.json"
for _d in (CKPTS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Silence harmless third-party deprecation warnings in the terminal.
warnings.filterwarnings(
    "ignore",
    message=r".*TRANSFORMERS_CACHE.*deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.nn\.utils\.weight_norm.*deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*weight_norm.*deprecated.*",
    category=FutureWarning,
)

# Keep Hugging Face/Xet noise quiet. hf_xet can be very slow on some Windows PCs,
# so disable Xet and fall back to normal HTTP downloads without printing Xet warnings.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
logging.getLogger("huggingface_hub.file_download").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)

# Force Hugging Face downloads/cache into the local ckpts folder, but preserve Hugging Face auth.
# Gated Stability AI repos require an HF token. When HF_HOME is moved to ./ckpts,
# huggingface_hub stops seeing the user's normal token unless we copy/read it first.
def _configure_huggingface_cache_and_auth() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    # Optional portable token file beside app.py. Do not include this file when redistributing publicly.
    token_file = APP_DIR / "hf_token.txt"
    if not token and token_file.exists():
        token = token_file.read_text(encoding="utf-8").strip()

    # Try the user's existing Hugging Face login token before changing HF_HOME.
    default_hf_home = Path.home() / ".cache" / "huggingface"
    old_token_file = default_hf_home / "token"
    if not token and old_token_file.exists():
        token = old_token_file.read_text(encoding="utf-8").strip()

    os.environ.setdefault("HF_HOME", str(CKPTS_DIR))
    os.environ.setdefault("HF_HUB_CACHE", str(CKPTS_DIR / "hub"))
    # Transformers >=4.56 warns if TRANSFORMERS_CACHE is present; HF_HOME/HF_HUB_CACHE are enough.
    os.environ.pop("TRANSFORMERS_CACHE", None)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        local_token_file = CKPTS_DIR / "token"
        try:
            local_token_file.write_text(token, encoding="utf-8")
        except Exception as exc:
            print(f"[auth] warning: could not write local Hugging Face token: {exc}", flush=True)


_configure_huggingface_cache_and_auth()

def _ensure_stable_audio_tools() -> None:
    try:
        import stable_audio_tools  # noqa: F401
        return
    except ImportError:
        pass
    # stable-audio-tools 0.0.20 strict-pins torch==2.7.1 / torchaudio==2.7.1,
    # which lack sm_120 (Blackwell) kernels. Install with --no-deps; the
    # transitive deps are listed in requirements.txt and resolved against the
    # sm_120-capable torch at build time.
    print("[startup] installing stable-audio-tools (--no-deps) …", flush=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
         "stable-audio-tools"],
    )
    import stable_audio_tools  # noqa: F401
    print("[startup] stable-audio-tools installed.", flush=True)

_ensure_stable_audio_tools()


import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
from einops import rearrange
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond_inpaint


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------


@dataclass
class Variant:
    key: str
    repo: str
    label: str
    default_duration: int
    max_seconds: int
    placeholder: str


VARIANTS: list[Variant] = [
    Variant(
        key="small-music",
        repo="stabilityai/stable-audio-3-small-music",
        label="Small Music — 0.6B, Music-Focused (4GB VRAM)",
        default_duration=60,
        max_seconds=120,
        placeholder="Cinematic neo-soul groove with electric piano, brushed drums, walking upright bass, smoky vibe 92 BPM",
    ),
    Variant(
        key="small-sfx",
        repo="stabilityai/stable-audio-3-small-sfx",
        label="Small SFX — 0.6B, Sound Effects (4GB VRAM)",
        default_duration=7,
        max_seconds=120,
        placeholder="Chugging train coming into station with horn",
    ),
    Variant(
        key="medium",
        repo="stabilityai/stable-audio-3-medium",
        label="Medium — General Audio (8GB-12GB VRAM)",
        default_duration=60,
        max_seconds=380,
        placeholder="A dream-like Synthpop instrumental that would accompany a dream-sequence in a surrealist movie 120 BPM",
    ),
    
]
VARIANT_MAP = {v.key: v for v in VARIANTS}
DEFAULT_VARIANT_KEY = "small-music"


# ---------------------------------------------------------------------------
# Lazy model loading: only one variant is kept in VRAM at a time
# ---------------------------------------------------------------------------

@dataclass
class LoadedVariant:
    variant: Variant
    model: object
    sample_rate: int
    sample_size: int
    max_seconds: int


LOADED: dict[str, LoadedVariant] = {}
CURRENT_VARIANT_KEY: Optional[str] = None


def _load_settings() -> dict:
    defaults = {
        "active_tab": "simple",
        "simple": {},
        "advanced": {},
    }
    if SETTINGS_PATH.exists():
        try:
            loaded = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                defaults.update(loaded)
        except Exception as e:
            print(f"[settings] could not load {SETTINGS_PATH.name}: {e}", flush=True)
    return defaults


def _save_settings(settings: dict) -> None:
    try:
        SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[settings] could not save {SETTINGS_PATH.name}: {e}", flush=True)


SETTINGS = _load_settings()


def _settings_for(tab: str, variant_key: str) -> dict:
    return SETTINGS.setdefault(tab, {}).setdefault(variant_key, {})


def _unload_models_except(variant_key: str) -> None:
    global CURRENT_VARIANT_KEY
    for key in list(LOADED.keys()):
        if key != variant_key:
            print(f"[models] unloading {key} from VRAM", flush=True)
            try:
                del LOADED[key].model
            except Exception:
                pass
            del LOADED[key]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    CURRENT_VARIANT_KEY = variant_key if variant_key in LOADED else None


def _load_variant(variant_key: str, progress: Optional[gr.Progress] = None) -> tuple[LoadedVariant, float, bool]:
    """Download/load the selected variant on demand.

    Only one model is kept in VRAM. Download failures are shown inside Gradio
    instead of crashing the whole app.
    """
    global CURRENT_VARIANT_KEY
    if variant_key not in VARIANT_MAP:
        raise gr.Error(f"Unknown variant {variant_key!r}.")
    if variant_key in LOADED:
        _unload_models_except(variant_key)
        return LOADED[variant_key], 0.0, False

    _unload_models_except(variant_key)
    v = VARIANT_MAP[variant_key]
    msg = f"[models] downloading/loading {v.repo} into VRAM/cache {CKPTS_DIR} …"
    print(msg, flush=True)
    if progress is not None:
        progress(0.02, desc=f"Downloading/loading {variant_key}. Check terminal for downloading/loading progress.")
    t0 = time.time()
    try:
        model, config = get_pretrained_model(v.repo)
    except Exception as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise gr.Error(
            "Model download/load failed.\n\n"
            "Check your internet connection, Hugging Face login/token, and that "
            "you accepted the Stability AI model license on Hugging Face.\n\n"
            f"Repo: {v.repo}\n\nOriginal error: {exc}"
        ) from exc

    sr = int(config["sample_rate"])
    ss = int(config["sample_size"])
    model = model.to("cuda").to(torch.float16)
    lv = LoadedVariant(
        variant=v,
        model=model,
        sample_rate=sr,
        sample_size=ss,
        max_seconds=ss // sr,
    )
    LOADED[variant_key] = lv
    CURRENT_VARIANT_KEY = variant_key
    load_time = time.time() - t0
    print(
        f"[models] {variant_key} ready in {load_time:.1f}s · "
        f"sr={sr} · sample_size={ss} (~{ss // sr}s max)",
        flush=True,
    )
    return lv, load_time, True

def _default_value(tab: str, variant_key: str, name: str, fallback):
    return _settings_for(tab, variant_key).get(name, fallback)


# Do not load/download any model at startup. The UI opens immediately; the
# selected model downloads/loads only when Generate is clicked.
print("[startup] no model loaded; models will download/load on first Generation", flush=True)

VARIANT_CHOICES = [(v.label, v.key) for v in VARIANTS]
# Samplers valid for rf_denoiser diffusion objective (the SA3 family).
SAMPLERS = ["pingpong", "euler", "rk4", "dpmpp"]
OUTPUT_FORMATS = ["wav", "mp3"]


# ---------------------------------------------------------------------------
# Spectrogram helper (Mel; adapted from the reference repo's aeiou.py)
# ---------------------------------------------------------------------------


def _power_to_db(spec: np.ndarray, amin: float = 1e-10) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(amin, spec))


def audio_spectrogram_image(
    waveform: torch.Tensor,
    sample_rate: int,
    db_range=(35, 120),
    figsize=(5, 4),
) -> Image.Image:
    """Render a Mel spectrogram (left channel) as a PIL image."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    n_fft = 1024
    hop_length = n_fft // 2
    mel_op = T.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, win_length=None,
        hop_length=hop_length, center=True, pad_mode="reflect", power=2.0,
        norm="slaney", onesided=True, n_mels=128, mel_scale="htk",
    )
    melspec = mel_op(waveform.float())[0]  # left channel
    fig = Figure(figsize=figsize, dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ax.imshow(_power_to_db(melspec.numpy()), origin="lower", aspect="auto",
              vmin=db_range[0], vmax=db_range[1])
    ax.set_ylabel("mel bins (log freq)")
    ax.set_xlabel("frame")
    ax.set_title("MelSpectrogram")
    canvas.draw()
    return Image.fromarray(np.asarray(canvas.buffer_rgba()))


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _gradio_audio_to_tensor(
    audio_in: Optional[Tuple[int, np.ndarray]],
) -> Optional[Tuple[int, torch.Tensor]]:
    """Convert a gr.Audio (numpy) value to the (sr, torch.Tensor[C,N]) tuple
    that ``generate_diffusion_cond_inpaint`` expects. Accepts mono or stereo."""
    if audio_in is None:
        return None
    sr, arr = audio_in
    if arr is None or (hasattr(arr, "size") and arr.size == 0):
        return None
    arr = np.asarray(arr)
    if arr.dtype.kind in ("i", "u"):
        max_val = float(np.iinfo(arr.dtype).max)
        arr = arr.astype(np.float32) / max_val
    else:
        arr = arr.astype(np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]                       # (1, N)
    else:
        # gr.Audio returns (N, C); transpose to (C, N)
        arr = arr.T if arr.shape[0] > arr.shape[1] else arr
    return int(sr), torch.from_numpy(arr)


def _safe_prompt_stem(prompt: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", prompt or "")[:3]
    return "_".join(words).lower() if words else "stable_audio"


def _tensor_to_audio_file(
    output: torch.Tensor,
    sample_rate: int,
    duration_seconds: Optional[int],
    prompt: str,
    seed: int,
    variant_key: str,
    output_format: str = "wav",
) -> Tuple[str, torch.Tensor]:
    """Pack a (B, C, N) generation tensor, trim first, normalise after trim,
    write to outputs/, and return (path, int16-tensor).

    Important: small SA3 models can contain peaks outside the requested duration.
    If we normalise before trimming, the kept audio may sound very quiet. So the
    cut happens before peak normalisation.
    """
    audio = rearrange(output, "b d n -> d (b n)").to(torch.float32).cpu()
    if duration_seconds is not None:
        audio = audio[:, : int(duration_seconds) * sample_rate]

    # Remove any DC offset, then peak-normalise the actual saved region.
    audio = audio - audio.mean(dim=-1, keepdim=True)
    peak = torch.max(torch.abs(audio)).clamp(min=1e-9)
    audio = (audio / peak * 0.90).clamp(-1, 1)

    # Perceptual boost for low-RMS outputs, common with the small models.
    rms = torch.sqrt(torch.mean(audio.float() ** 2)).item()
    target_rms = 0.09
    if 0 < rms < target_rms:
        # Conservative boost to avoid clipping/harsh distortion on small models.
        audio = (audio * min(target_rms / rms, 2.0)).clamp(-0.95, 0.95)

    # Final safety peak limiter before saving.
    final_peak = torch.max(torch.abs(audio)).clamp(min=1e-9)
    if final_peak > 0.95:
        audio = audio / final_peak * 0.95

    int16_audio = audio.mul(32767).round().to(torch.int16)

    # Save each model's generations into its own folder:
    # outputs/small-music, outputs/medium, outputs/small-sfx
    safe_variant = re.sub(r"[^A-Za-z0-9_.-]+", "_", variant_key or "unknown-model").strip("._") or "unknown-model"
    model_output_dir = OUTPUTS_DIR / safe_variant
    model_output_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    fmt = (output_format or "wav").lower().strip()
    if fmt not in OUTPUT_FORMATS:
        fmt = "wav"
    out_path = model_output_dir / f"{stamp}_{_safe_prompt_stem(prompt)}_seed-{int(seed)}.{fmt}"   

    float_data = audio.numpy().T.astype(np.float32)
    int16_data = int16_audio.numpy().T
    if fmt == "mp3":
        try:
            sf.write(str(out_path), float_data, sample_rate, format="MP3")
        except Exception as e:
            raise gr.Error(
                "MP3 export failed. Your soundfile/libsndfile build may not include MP3 encoding. "
                "Choose WAV or bundle an MP3-capable libsndfile/ffmpeg with your app. "
                f"Original error: {e}"
            )
    else:
        sf.write(str(out_path), int16_data, sample_rate, subtype="PCM_16")
    return str(out_path), int16_audio

def _remember_generation_settings(tab: str, variant_key: str, values: dict) -> None:
    SETTINGS["active_tab"] = tab
    SETTINGS.setdefault(tab, {})[variant_key] = values
    _save_settings(SETTINGS)


def _run_inference(
    variant_key: str,
    prompt: str,
    negative_prompt: str = "",
    duration: int = 60,
    steps: int = 8,
    cfg_scale: float = 1.0,
    sampler_type: str = "pingpong",
    seed: int = 0,
    random_seed: bool = True,
    sigma_max: float = 1.0,
    apg_scale: float = 1.0,
    duration_padding_sec: float = 6.0,
    cut_to_seconds_total: bool = True,
    init_audio: Optional[Tuple[int, np.ndarray]] = None,
    init_noise_level: float = 0.9,
    inpaint_audio: Optional[Tuple[int, np.ndarray]] = None,
    mask_start_sec: float = 0.0,
    mask_end_sec: float = 0.0,
    preview_every: int = 0,
    return_spectrogram: bool = True,
    output_format: str = "wav",
    progress: gr.Progress = gr.Progress(),
):
    """Full-featured generation. Returns (audio_path, [spectrogram_img, *previews])
    when ``return_spectrogram`` is True, else just ``audio_path``."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Please enter a prompt.")
    total_t0 = time.time()
    lv, load_time, did_load = _load_variant(variant_key, progress=progress)
    duration = max(1, min(int(duration), lv.max_seconds))

    progress(0.05, desc=f"[{variant_key}] preparing conditioning")
    conditioning = [{"prompt": prompt, "seconds_total": int(duration)}]
    negative_conditioning = None
    neg = (negative_prompt or "").strip()
    if neg:
        negative_conditioning = [{"prompt": neg, "seconds_total": int(duration)}]

    # The pretransform encoder is fp16 (we cast the whole model at startup),
    # but prepare_audio's torchaudio Resample uses an fp32 kernel. Pre-resample
    # in fp32 here so prepare_audio's resample is a no-op, then cast to the
    # model dtype so the encoder doesn't see a dtype mismatch.
    model_dtype = next(lv.model.parameters()).dtype

    def _prep(tup):
        if tup is None:
            return None
        sr, t = tup
        t = t.float()
        if sr != lv.sample_rate:
            t = torchaudio.functional.resample(t, sr, lv.sample_rate)
        return lv.sample_rate, t.to(model_dtype)

    init_audio_t = _prep(_gradio_audio_to_tensor(init_audio))
    inpaint_audio_t = _prep(_gradio_audio_to_tensor(inpaint_audio))

    # Inpaint mask: only enable if mask_end > mask_start AND we have either
    # inpaint_audio or init_audio (otherwise the mask wraps zero content).
    mask_start = max(0.0, float(mask_start_sec))
    mask_end = min(float(duration), float(mask_end_sec))
    use_mask = (
        inpaint_audio_t is not None
        and mask_end > mask_start
    )

    if random_seed:
        seed_val = random.randint(1, 2**31 - 1)
    else:
        seed_val = int(seed)
        seed_val = max(0, min(seed_val, 2**31 - 1))

    preview_images: list = []
    callback = None
    if preview_every and int(preview_every) > 0:
        every = int(preview_every)

        def _cb(info):
            i = info["i"]
            if i % every != 0:
                return
            denoised = info["denoised"]
            try:
                if lv.model.pretransform is not None:
                    denoised = lv.model.pretransform.decode(denoised)
                d = rearrange(denoised, "b d n -> d (b n)")
                d = d.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                img = audio_spectrogram_image(d, sample_rate=lv.sample_rate)
                preview_images.append((img, f"Step {i + 1}"))
            except Exception as e:
                print(f"[preview] skipped step {i}: {e}", flush=True)
        callback = _cb

    gen_kwargs: dict = dict(
        steps=int(steps),
        cfg_scale=float(cfg_scale),
        conditioning=conditioning,
        negative_conditioning=negative_conditioning,
        sample_size=lv.sample_size,
        sampler_type=sampler_type,
        seed=seed_val,
        device="cuda",
        sigma_max=float(sigma_max),
        apg_scale=float(apg_scale),
        duration_padding_sec=float(duration_padding_sec),
    )
    if init_audio_t is not None:
        gen_kwargs["init_audio"] = init_audio_t
        gen_kwargs["init_noise_level"] = float(init_noise_level)
    if inpaint_audio_t is not None:
        gen_kwargs["inpaint_audio"] = inpaint_audio_t
    if use_mask:
        gen_kwargs["inpaint_mask_start_seconds"] = mask_start
        gen_kwargs["inpaint_mask_end_seconds"] = mask_end
    if callback is not None:
        gen_kwargs["callback"] = callback

    progress(0.25, desc=f"[{variant_key}] sampling {steps} steps with {sampler_type}")
    t0 = time.time()
    output = generate_diffusion_cond_inpaint(lv.model, **gen_kwargs)
    gen_time = time.time() - t0
    print(f"[infer/{variant_key}] sampling done in {gen_time:.1f}s", flush=True)

    progress(0.92, desc="Normalising & saving")
    cut_dur = int(duration) if cut_to_seconds_total else None
    out_path, int16_audio = _tensor_to_audio_file(output, lv.sample_rate, cut_dur, prompt, seed_val, variant_key, output_format)
    total_time = time.time() - total_t0
    load_msg = f"Loaded/downloaded model in {load_time:.1f}s. " if did_load else "Model already loaded. "
    status = (
        f"{load_msg}Generated in {gen_time:.1f}s. Total: {total_time:.1f}s. "
        f"Seed: {seed_val}. Saved: {out_path}"
    )
    print(f"[done/{variant_key}] {status}", flush=True)

    if not return_spectrogram:
        return out_path, seed_val, status

    spec_img = audio_spectrogram_image(int16_audio, sample_rate=lv.sample_rate)
    return out_path, [spec_img, *preview_images], seed_val, status


@spaces.GPU
def infer(
    variant_key: str,
    prompt: str,
    duration: int = 60,
    steps: int = 8,
    cfg_scale: float = 1.0,
    sampler_type: str = "pingpong",
    seed: int = 0,
    random_seed: bool = True,
    output_format: str = "wav",
    progress: gr.Progress = gr.Progress(),
):
    """Slim handler used by the Simple tab and the Examples cache."""
    out_path, used_seed, status = _run_inference(
        variant_key=variant_key,
        prompt=prompt,
        duration=duration,
        steps=steps,
        cfg_scale=cfg_scale,
        sampler_type=sampler_type,
        seed=seed,
        random_seed=random_seed,
        output_format=output_format,
        return_spectrogram=False,
        progress=progress,
    )
    _remember_generation_settings("simple", variant_key, {"prompt": prompt, "duration": duration, "steps": steps, "cfg_scale": cfg_scale, "sampler_type": sampler_type, "seed": used_seed, "random_seed": random_seed, "output_format": output_format})
    return out_path, used_seed, status


@spaces.GPU
def infer_advanced(
    variant_key: str,
    prompt: str,
    negative_prompt: str,
    duration: int,
    steps: int,
    cfg_scale: float,
    sampler_type: str,
    seed: int,
    random_seed: bool,
    sigma_max: float,
    apg_scale: float,
    duration_padding_sec: float,
    cut_to_seconds_total: bool,
    init_audio: Optional[Tuple[int, np.ndarray]],
    init_noise_level: float,
    inpaint_audio: Optional[Tuple[int, np.ndarray]],
    mask_start_sec: float,
    mask_end_sec: float,
    preview_every: int,
    output_format: str = "wav",
    progress: gr.Progress = gr.Progress(),
):
    """Full-featured handler used by the Advanced tab."""
    out_path, gallery, used_seed, status = _run_inference(
        variant_key=variant_key,
        prompt=prompt,
        negative_prompt=negative_prompt,
        duration=duration,
        steps=steps,
        cfg_scale=cfg_scale,
        sampler_type=sampler_type,
        seed=seed,
        random_seed=random_seed,
        sigma_max=sigma_max,
        apg_scale=apg_scale,
        duration_padding_sec=duration_padding_sec,
        cut_to_seconds_total=cut_to_seconds_total,
        init_audio=init_audio,
        init_noise_level=init_noise_level,
        inpaint_audio=inpaint_audio,
        mask_start_sec=mask_start_sec,
        mask_end_sec=mask_end_sec,
        preview_every=preview_every,
        output_format=output_format,
        return_spectrogram=True,
        progress=progress,
    )
    _remember_generation_settings("advanced", variant_key, {"prompt": prompt, "negative_prompt": negative_prompt, "duration": duration, "steps": steps, "cfg_scale": cfg_scale, "sampler_type": sampler_type, "seed": used_seed, "random_seed": random_seed, "sigma_max": sigma_max, "apg_scale": apg_scale, "duration_padding_sec": duration_padding_sec, "cut_to_seconds_total": cut_to_seconds_total, "init_noise_level": init_noise_level, "mask_start_sec": mask_start_sec, "mask_end_sec": mask_end_sec, "preview_every": preview_every, "output_format": output_format})
    return out_path, gallery, used_seed, status


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

DESCRIPTION = """
# 🎵 Stable Audio 3 

Pick a model, write a prompt, hit Generate, the models will automatically download upon first use . 

Switch to **Advanced Tab** for the full sampler / init-audio / inpainting controls.

Need help in Prompting : <a href="https://github.com/Stability-AI/stable-audio-3/blob/main/docs/guides/prompting.md" target="_blank" rel="noopener noreferrer">Prompting Guide</a>

Need to report a problem: <a href="https://huggingface.co/spaces/stabilityai/stable-audio-3/discussions" target="_blank" rel="noopener noreferrer">Stable Audio 3</a>
"""



def _variant_change_simple(variant_key: str):
    v = VARIANT_MAP[variant_key]
    st = _settings_for("simple", variant_key)
    dur = int(st.get("duration", min(v.default_duration, v.max_seconds)))
    dur = max(1, min(dur, v.max_seconds))
    return (
        gr.update(maximum=v.max_seconds, value=dur,
                  label=f"Duration (s) · model max {v.max_seconds}s"),
        gr.update(placeholder=v.placeholder, value=st.get("prompt", "")),
        gr.update(value=st.get("steps", 8)),
        gr.update(value=st.get("cfg_scale", 1.0)),
        gr.update(value=st.get("sampler_type", "pingpong")),
        gr.update(value=st.get("seed", 0)),
        gr.update(value=st.get("random_seed", True)),
        gr.update(value=st.get("output_format", "wav")),
    )

def _variant_change_advanced(variant_key: str):
    v = VARIANT_MAP[variant_key]
    st = _settings_for("advanced", variant_key)
    dur = int(st.get("duration", min(v.default_duration, v.max_seconds)))
    dur = max(1, min(dur, v.max_seconds))
    return (
        gr.update(maximum=v.max_seconds, value=dur,
                  label=f"Seconds total · model max {v.max_seconds}s"),
        gr.update(placeholder=v.placeholder, value=st.get("prompt", "")),
        gr.update(value=st.get("negative_prompt", "")),
        gr.update(value=st.get("steps", 8)),
        gr.update(value=st.get("cfg_scale", 1.0)),
        gr.update(value=st.get("sampler_type", "pingpong")),
        gr.update(value=st.get("seed", 0)),
        gr.update(value=st.get("random_seed", True)),
        gr.update(value=st.get("sigma_max", 1.0)),
        gr.update(value=st.get("apg_scale", 1.0)),
        gr.update(value=st.get("duration_padding_sec", 6.0)),
        gr.update(value=st.get("cut_to_seconds_total", True)),
        gr.update(value=st.get("init_noise_level", 0.9)),
        gr.update(maximum=float(v.max_seconds), value=float(st.get("mask_start_sec", 0.0))),
        gr.update(maximum=float(v.max_seconds), value=float(st.get("mask_end_sec", dur))),
        gr.update(value=st.get("preview_every", 0)),
        gr.update(value=st.get("output_format", "wav")),
    )





with gr.Blocks(title="Stable Audio 3") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tabs():
        # -----------------------------------------------------------------
        # Simple tab
        # -----------------------------------------------------------------
        with gr.Tab("Simple"):
            variant = gr.Radio(
                choices=VARIANT_CHOICES,
                value=DEFAULT_VARIANT_KEY,
                label="Model",
            )

            with gr.Row():
                with gr.Column(scale=2):
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder=VARIANT_MAP[DEFAULT_VARIANT_KEY].placeholder,
                        value=_default_value("simple", DEFAULT_VARIANT_KEY, "prompt", ""),
                        lines=3,
                    )
                    duration = gr.Slider(
                        1, VARIANT_MAP[DEFAULT_VARIANT_KEY].max_seconds,
                        value=_default_value("simple", DEFAULT_VARIANT_KEY, "duration", VARIANT_MAP[DEFAULT_VARIANT_KEY].default_duration), step=1,
                        label=f"Duration (s) · model max {VARIANT_MAP[DEFAULT_VARIANT_KEY].max_seconds}s",
                    )
                    with gr.Accordion("Advanced settings", open=False):
                        steps = gr.Slider(1, 50, value=_default_value("simple", DEFAULT_VARIANT_KEY, "steps", 8), step=1, label="Steps")
                        cfg_scale = gr.Slider(0.5, 8.0, value=_default_value("simple", DEFAULT_VARIANT_KEY, "cfg_scale", 1.0), step=0.1, label="CFG scale")
                        sampler_type = gr.Dropdown(SAMPLERS, value=_default_value("simple", DEFAULT_VARIANT_KEY, "sampler_type", "pingpong"), label="Sampler")
                        with gr.Row():
                            seed = gr.Number(value=_default_value("simple", DEFAULT_VARIANT_KEY, "seed", 0), precision=0, label="Seed")
                            random_seed = gr.Checkbox(value=_default_value("simple", DEFAULT_VARIANT_KEY, "random_seed", True), label="Random Seed")
                        output_format = gr.Radio(OUTPUT_FORMATS, value=_default_value("simple", DEFAULT_VARIANT_KEY, "output_format", "wav"), label="Save as")

                with gr.Column(scale=1):
                    audio_out = gr.Audio(label="Output", type="filepath", autoplay=True)
                    run_btn = gr.Button("🎼 Generate", variant="primary", size="lg")
                    status = gr.Textbox(label="Status", value="Ready. Model will download/load when you click Generate.", lines=4, interactive=False, elem_classes=["status-box"])

            variant.change(
                fn=_variant_change_simple,
                inputs=[variant],
                outputs=[duration, prompt, steps, cfg_scale, sampler_type, seed, random_seed, output_format],
            )

            run_btn.click(
                fn=infer,
                inputs=[variant, prompt, duration, steps, cfg_scale, sampler_type, seed, random_seed, output_format],
                outputs=[audio_out, seed, status],
            )

        # -----------------------------------------------------------------
        # Advanced tab — mirrors stable_audio_3/interface/diffusion_cond.py
        # -----------------------------------------------------------------
        with gr.Tab("Advanced"):
            adv_variant = gr.Radio(
                choices=VARIANT_CHOICES,
                value=DEFAULT_VARIANT_KEY,
                label="Model",
            )

            with gr.Row():
                with gr.Column(scale=6):
                    adv_prompt = gr.Textbox(
                        show_label=False,
                        placeholder=VARIANT_MAP[DEFAULT_VARIANT_KEY].placeholder,
                        value=_default_value("advanced", DEFAULT_VARIANT_KEY, "prompt", ""),
                    )
                    adv_negative = gr.Textbox(
                        show_label=False, placeholder="Negative prompt",
                        value=_default_value("advanced", DEFAULT_VARIANT_KEY, "negative_prompt", ""),
                    )

            with gr.Row(equal_height=False):
                with gr.Column():
                    adv_seconds_total = gr.Slider(
                        minimum=1,
                        maximum=VARIANT_MAP[DEFAULT_VARIANT_KEY].max_seconds,
                        step=1,
                        value=_default_value("advanced", DEFAULT_VARIANT_KEY, "duration", VARIANT_MAP[DEFAULT_VARIANT_KEY].default_duration),
                        label=f"Seconds total · model max {VARIANT_MAP[DEFAULT_VARIANT_KEY].max_seconds}s",
                    )

                    with gr.Row():
                        adv_steps = gr.Slider(
                            minimum=1, maximum=500, step=1, value=_default_value("advanced", DEFAULT_VARIANT_KEY, "steps", 8), label="Steps"
                        )
                        adv_cfg = gr.Slider(
                            minimum=0.0, maximum=25.0, step=0.1, value=_default_value("advanced", DEFAULT_VARIANT_KEY, "cfg_scale", 1.0),
                            label="CFG scale",
                        )

                    with gr.Accordion("Sampler params", open=False):
                        with gr.Row():
                            with gr.Column():
                                adv_seed = gr.Number(
                                    label="Seed",
                                    value=_default_value("advanced", DEFAULT_VARIANT_KEY, "seed", 0), precision=0,
                                )
                                adv_random_seed = gr.Checkbox(value=_default_value("advanced", DEFAULT_VARIANT_KEY, "random_seed", True), label="Random Seed")
                            adv_sampler = gr.Dropdown(
                                SAMPLERS, label="Sampler type", value=_default_value("advanced", DEFAULT_VARIANT_KEY, "sampler_type", "pingpong"),
                            )
                            adv_sigma_max = gr.Slider(
                                minimum=0.0, maximum=1.0, step=0.01, value=_default_value("advanced", DEFAULT_VARIANT_KEY, "sigma_max", 1.0),
                                label="Sigma max",
                            )
                        with gr.Row():
                            adv_apg = gr.Slider(
                                minimum=0.0, maximum=1.0, step=0.1, value=_default_value("advanced", DEFAULT_VARIANT_KEY, "apg_scale", 1.0),
                                label="APG scale", info="1.0=full APG, 0.0=vanilla CFG",
                            )
                            adv_dur_padding = gr.Slider(
                                minimum=0.0, maximum=30.0, step=0.5, value=_default_value("advanced", DEFAULT_VARIANT_KEY, "duration_padding_sec", 6.0),
                                label="Duration padding (sec)",
                            )

                    with gr.Accordion("Output params", open=False):
                        with gr.Row():
                            adv_preview_every = gr.Slider(
                                minimum=0, maximum=100, step=1, value=_default_value("advanced", DEFAULT_VARIANT_KEY, "preview_every", 0),
                                label="Spec preview every N steps (0 = off)",
                            )
                            adv_cut_to_total = gr.Checkbox(
                                label="Cut to seconds total", value=_default_value("advanced", DEFAULT_VARIANT_KEY, "cut_to_seconds_total", True),
                            )
                            adv_output_format = gr.Radio(OUTPUT_FORMATS, value=_default_value("advanced", DEFAULT_VARIANT_KEY, "output_format", "wav"), label="Save as")

                    with gr.Accordion("Init audio", open=False):
                        adv_init_audio = gr.Audio(
                            label="Init audio",
                            type="numpy",
                        )
                        adv_init_noise = gr.Slider(
                            minimum=0.01, maximum=1.0, step=0.01, value=_default_value("advanced", DEFAULT_VARIANT_KEY, "init_noise_level", 0.9),
                            label="Init noise level",
                        )

                    with gr.Accordion("Inpainting", open=False):
                        adv_inpaint_audio = gr.Audio(
                            label="Inpaint audio",
                            type="numpy",
                        )
                        adv_mask_start = gr.Slider(
                            minimum=0.0,
                            maximum=float(VARIANT_MAP[DEFAULT_VARIANT_KEY].max_seconds),
                            step=0.1, value=_default_value("advanced", DEFAULT_VARIANT_KEY, "mask_start_sec", 0.0), label="Mask start (sec)",
                        )
                        adv_mask_end = gr.Slider(
                            minimum=0.0,
                            maximum=float(VARIANT_MAP[DEFAULT_VARIANT_KEY].max_seconds),
                            step=0.1, value=_default_value("advanced", DEFAULT_VARIANT_KEY, "mask_end_sec", 0.0), label="Mask end (sec)",
                        )

                with gr.Column():
                    adv_audio_out = gr.Audio(
                        label="Output audio", type="filepath", autoplay=False,
                        sources=[],
                    )
                    adv_generate = gr.Button("Generate", variant="primary", size="lg")
                    adv_status = gr.Textbox(label="Status", value="Ready. Model will download/load when you click Generate.", lines=4, interactive=False, elem_classes=["status-box"])
                    adv_spec_gallery = gr.Gallery(
                        label="Output spectrogram", show_label=True, columns=2,
                    )
                    send_to_init_btn = gr.Button("Send to init audio")
                    send_to_inpaint_btn = gr.Button("Send to inpaint audio")

            send_to_init_btn.click(
                fn=lambda a: a, inputs=[adv_audio_out], outputs=[adv_init_audio]
            )
            send_to_inpaint_btn.click(
                fn=lambda a: a, inputs=[adv_audio_out], outputs=[adv_inpaint_audio]
            )

            # Keep the inpaint mask bounded by the current duration.
            def _update_mask_max(seconds_total):
                m = max(float(seconds_total), 1.0)
                return (
                    gr.update(maximum=m),
                    gr.update(maximum=m, value=m),
                )
            adv_seconds_total.change(
                _update_mask_max,
                inputs=[adv_seconds_total],
                outputs=[adv_mask_start, adv_mask_end],
            )

            adv_variant.change(
                fn=_variant_change_advanced,
                inputs=[adv_variant],
                outputs=[adv_seconds_total, adv_prompt, adv_negative, adv_steps, adv_cfg, adv_sampler, adv_seed, adv_random_seed, adv_sigma_max, adv_apg, adv_dur_padding, adv_cut_to_total, adv_init_noise, adv_mask_start, adv_mask_end, adv_preview_every, adv_output_format],
            )

            adv_generate.click(
                fn=infer_advanced,
                inputs=[
                    adv_variant,
                    adv_prompt,
                    adv_negative,
                    adv_seconds_total,
                    adv_steps,
                    adv_cfg,
                    adv_sampler,
                    adv_seed,
                    adv_random_seed,
                    adv_sigma_max,
                    adv_apg,
                    adv_dur_padding,
                    adv_cut_to_total,
                    adv_init_audio,
                    adv_init_noise,
                    adv_inpaint_audio,
                    adv_mask_start,
                    adv_mask_end,
                    adv_preview_every,
                    adv_output_format,
                ],
                outputs=[adv_audio_out, adv_spec_gallery, adv_seed, adv_status],
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inbrowser", action="store_true", help="Open the Gradio UI in the default browser after startup.")
    args, _ = parser.parse_known_args()
    demo.launch(inbrowser=args.inbrowser)
