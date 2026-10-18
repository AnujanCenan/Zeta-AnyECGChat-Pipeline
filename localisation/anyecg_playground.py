#!/usr/bin/env python3
"""
anyecg_playground.py -- a self-contained sandbox for anyECG-chat.

Lets you load an ECG and interactively try the two features you care about:

  * report generation   ->  "Report: Sinus rhythm, Poor R wave progression ..."
  * waveform localisation -> "Duration: 1.9s-3.1s, 6.8s-8.1s"  (+ an overlay plot)

Nothing in the anyECG-chat repo is edited or required to be next to this script.
The repo is found via ANTECG_REPO in your .env and added with sys.path.append.
The repo's hard-coded author paths (/mnt/sda1/xxxx/..., /home/xxxx/...) are
bypassed by subclassing ECG_Language_Model and overriding the two loader methods.

--------------------------------------------------------------------------
Quick start
--------------------------------------------------------------------------
  1. cp .env.example .env      and fill it in (see comments in that file)
  2. pip install torch transformers peft accelerate einops wfdb scipy numpy matplotlib
     (the repo pins transformers==4.48.0, torch==2.5.1, numpy==2.2.0; its requirements.txt
      omits einops/peft/accelerate/scipy/matplotlib which are needed)
  3. python anyecg_playground.py                         # interactive prompt
     python anyecg_playground.py --ecg /data/mitdb/100 --report --locate pvc
     python anyecg_playground.py --check                 # validate .env, no GPU needed
     python anyecg_playground.py --ecg /data/mitdb/100 --no-model   # test ECG load + plot only

Interactive commands (type `help` inside the prompt):
  load <record> [start_s [end_s]] [keep=ii,v1]   WFDB record base path (no extension), or .npy/.csv
  report                            generate the ECG report
  locate <name>                     pvc | lbbb | rbbb | pac | svpb | any free text
  ask <question>                    free-form question about the loaded ECG
  set temperature|samples|max_new_tokens|seed <value>
  lora on|off                       toggle the LoRA adapter (sanity-check it is really applied)
  plot | info | help | quit

Notes on what the model actually does (from the repo + paper):
  * Input is resampled to 100 Hz, each lead min-max scaled to [-1, 1], leads in standard
    order, missing leads zero-filled. ECGs longer than 10 s are cut into 10 s clips.
  * Localisation is pure text generation ("Duration: Xs-Ys"), times are relative to the START of
    the window you fed in, and it was only trained on beat-type names: PVC, LBBB, RBBB, atrial
    premature beat, supraventricular premature/ectopic beat (MIT-BIH / European ST-T, 2-lead).
    Other names are allowed here but are out-of-distribution.
  * There is NO attention/saliency map. With --samples K > 1 this script re-asks the model K times
    (temperature > 0) and shades each time point by the FRACTION of answers covering it. That is a
    sampling-agreement score, not model confidence or attention.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SCRIPT_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------------------------
# .env handling (tiny parser, so python-dotenv is not required)
# --------------------------------------------------------------------------------------

DEFAULT_LLM_ID = "meta-llama/Meta-Llama-3-8B-Instruct"


def _parse_env_file(path: Path) -> dict:
    out = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip()
        if val[:1] in ("'", '"') and val.find(val[0], 1) > 0:
            val = val[1:val.find(val[0], 1)]          # quoted: take text up to the closing quote
        else:  # unquoted: strip trailing inline comment
            val = re.split(r"\s+#", val, maxsplit=1)[0].strip()
        out[key] = val
    return out


def load_env(explicit: Optional[str]) -> Optional[Path]:
    """Load a .env (explicit path > ./.env > <script dir>/.env). Real environment variables win."""
    candidates = [Path(explicit)] if explicit else [Path.cwd() / ".env", SCRIPT_DIR / ".env"]
    for p in candidates:
        if p.is_file():
            for k, v in _parse_env_file(p).items():
                os.environ.setdefault(k, v)
            os.environ["_ENV_FILE_DIR"] = str(p.resolve().parent)
            return p.resolve()
    if explicit:
        sys.exit(f"[error] --env file not found: {explicit}")
    return None


def env_path(key: str, *aliases: str) -> Optional[str]:
    """Read a path from the env; relative paths are resolved against the .env file's folder."""
    for k in (key, *aliases):
        v = os.environ.get(k, "").strip()
        if v:
            v = os.path.expanduser(os.path.expandvars(v))
            if not os.path.isabs(v) and os.environ.get("_ENV_FILE_DIR"):
                cand = os.path.join(os.environ["_ENV_FILE_DIR"], v)
                if os.path.exists(cand):
                    return cand
            return v
    return None


@dataclass
class Config:
    repo: str
    projection_ckpt: str
    ecg_model_ckpt: str
    lora_ckpt: str
    ecg_encoder_ckpt: str
    llm_id: str
    hf_token: Optional[str]
    ecg_dir: Optional[str]
    output_dir: str
    temperature: float
    max_new_tokens: int


def build_config(need_model: bool) -> Config:
    repo = env_path("ANTECG_REPO", "ANYECG_REPO")
    projection = env_path("PROJECTION_CKPT")
    ecg_model = env_path("ECG_MODEL_CKPT")
    lora = env_path("LORA_CKPT")
    encoder = env_path("ECG_ENCODER_CKPT")
    llm = env_path("LLM_MODEL_ID") or DEFAULT_LLM_ID
    if lora is None and projection:  # HF stage3_ckpt/ keeps adapter + projection + ecg_model together
        lora = os.path.dirname(projection)

    problems = []
    if not repo:
        problems.append("ANTECG_REPO is not set (path to your clone of the anyECG-chat code repo)")
    elif not os.path.isdir(os.path.join(repo, "anyecg")):
        problems.append(f"ANTECG_REPO={repo} has no 'anyecg/' package inside it")

    if need_model:
        for key, val, kind in [
            ("PROJECTION_CKPT", projection, "file"),
            ("ECG_MODEL_CKPT", ecg_model, "file"),
            ("LORA_CKPT", lora, "dir"),
            ("ECG_ENCODER_CKPT", encoder, "file"),
        ]:
            if not val:
                problems.append(f"{key} is not set")
            elif kind == "file" and not os.path.isfile(val):
                problems.append(f"{key}={val} is not a file")
            elif kind == "dir" and not os.path.isfile(os.path.join(val, "adapter_config.json")):
                problems.append(f"{key}={val} must be a directory containing adapter_config.json "
                                f"+ adapter_model.safetensors")
    if problems:
        sys.exit("[error] .env problems:\n  - " + "\n  - ".join(problems)
                 + "\n(see .env.example; use --env to point at a specific file)")

    return Config(
        repo=repo, projection_ckpt=projection or "", ecg_model_ckpt=ecg_model or "",
        lora_ckpt=lora or "", ecg_encoder_ckpt=encoder or "", llm_id=llm,
        hf_token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None,
        ecg_dir=env_path("ECG_DIR"),
        output_dir=env_path("OUTPUT_DIR") or str(Path.cwd() / "anyecg_outputs"),
        temperature=float(os.environ.get("TEMPERATURE", "0.6")),
        max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", "256")),
    )


# --------------------------------------------------------------------------------------
# Constants taken from the repo / paper
# --------------------------------------------------------------------------------------

STD_LEADS = ["i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"]
MODEL_FS = 100            # Hz, fixed by the model
CLIP_SAMPLES = 1000       # 10 s @ 100 Hz = one encoder pass = 61 LLM tokens (CLS + 60 patches)
TOKENS_PER_CLIP = 60      # + 1 averaged CLS token for the whole ECG
PROMPT_LIMIT = 512        # apply_chat_template(max_length=512, truncation=True) in the repo
MAX_SECONDS = 70          # 7 clips = 421 ECG tokens: the most that fits under PROMPT_LIMIT with a short question

REPORT_QUESTION = "Please provide the report for the following ECG."  # exact string used in inference.py

# Phrases copied from the `question` field of data/location_evaluation/*.json (the training vocabulary).
ABNORMALITIES = {
    "pvc": "Premature ventricular contraction",
    "lbbb": "Left bundle branch block beat",
    "rbbb": "Right bundle branch block beat",
    "pac": "Atrial premature beat",
    "svpb": "Supraventricular premature or ectopic beat (atrial or nodal)",
}
ALIASES = {"v": "pvc", "vpc": "pvc", "l": "lbbb", "r": "rbbb", "a": "pac", "apb": "pac",
           "s": "svpb", "sv": "svpb", "svpc": "svpb"}
LOCATE_TEMPLATE = "Where is the {X} located in this ECG?"   # one of the ~20 phrasings seen in training


def resolve_abnormality(text: str) -> Tuple[str, bool]:
    """-> (phrase to send, is_in_training_vocabulary)."""
    key = text.strip().lower()
    key = ALIASES.get(key, key)
    if key in ABNORMALITIES:
        return ABNORMALITIES[key], True
    for phrase in ABNORMALITIES.values():
        if key == phrase.lower():
            return phrase, True
    return text.strip(), False


# --------------------------------------------------------------------------------------
# ECG loading / preprocessing (mirrors anyecg.utils.get_ecg_from_path + ecg_transform)
# --------------------------------------------------------------------------------------

@dataclass
class LoadedECG:
    name: str
    model_input: "object"        # np.ndarray (12, T) float32, normalised [-1, 1] -- what the model sees
    display: "object"            # np.ndarray (12, T) 100 Hz, un-normalised (for plotting)
    present: List[bool]          # which of the 12 slots held a real recorded lead
    source_fs: float
    window_s: Tuple[float, float]  # (start, end) seconds within the source recording
    units: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return self.model_input.shape[1] / MODEL_FS


def _canon_lead_names(names: Sequence[str]) -> List[str]:
    n = [s.strip().lower() for s in names]
    # same special cases as the repo's get_ecg_from_path
    if n == ["ecg", "ecg"]:
        n = ["mlii", "v1"]
    elif n == ["ecg"]:
        n = ["i"]
    return [s.replace("ml", "").replace("d3", "iii") for s in n]


def _slot_leads(sig: "object", names: Sequence[str]) -> Tuple["object", List[bool], List[str]]:
    """(T, n) signal + lead names -> (T, 12) in standard order with zero-filled gaps."""
    import numpy as np
    canon = _canon_lead_names(names)
    out = np.zeros((sig.shape[0], 12), dtype=np.float64)
    present, notes = [False] * 12, []
    for col, nm in enumerate(canon):
        if nm in STD_LEADS:
            slot = STD_LEADS.index(nm)
            if present[slot]:
                notes.append(f"duplicate lead '{nm}' ignored (channel {col})")
                continue
            out[:, slot] = sig[:, col]
            present[slot] = True
        else:
            notes.append(f"unrecognised lead name '{names[col]}' skipped (use leads=... to map it)")
    if not any(present):
        raise ValueError(f"None of the channel names {list(names)} map to a standard lead. "
                         f"Pass leads=i,ii,... to say which lead each channel is.")
    return out, present, notes


def load_ecg(spec: str, start_s: Optional[float], end_s: Optional[float], ecg_dir: Optional[str],
             fs: Optional[float] = None, leads: Optional[Sequence[str]] = None,
             default_seconds: float = 10.0, keep: Optional[Sequence[str]] = None) -> LoadedECG:
    import numpy as np
    from scipy.signal import resample
    from anyecg.utils import ecg_transform  # repo's own normaliser: per-lead min-max -> [-1, 1], NaN -> 0

    p = Path(os.path.expanduser(spec))
    if not p.is_absolute() and ecg_dir:
        p = Path(ecg_dir) / spec
    suffix = p.suffix.lower()
    notes: List[str] = []
    start_s = 0.0 if start_s is None else float(start_s)
    units = ""

    if suffix in (".npy", ".csv", ".txt"):
        arr = np.load(p) if suffix == ".npy" else np.loadtxt(p, delimiter=",")
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.shape[0] <= 12 and arr.shape[1] > arr.shape[0]:
            arr = arr.T                       # (leads, T) -> (T, leads)
        if fs is None:
            raise ValueError("Array input needs a sampling rate: `load file.npy fs=500` (or --fs 500).")
        names = list(leads) if leads else (STD_LEADS if arr.shape[1] == 12 else None)
        if names is None or len(names) != arr.shape[1]:
            raise ValueError(f"{arr.shape[1]} channels: pass leads=<names> with one name per channel.")
        a, b = int(round(start_s * fs)), (int(round(end_s * fs)) if end_s is not None
                                          else int(round((start_s + default_seconds) * fs)))
        sig, source_fs = arr[a:b].astype(np.float64), float(fs)
        name = p.stem
    else:
        import wfdb
        base = str(p.with_suffix("")) if suffix in (".hea", ".dat") else str(p)
        if not os.path.isfile(base + ".hea"):
            raise FileNotFoundError(f"No WFDB header found at {base}.hea  (give the record path WITHOUT extension)")
        hdr = wfdb.rdheader(base)
        source_fs = float(hdr.fs)
        a = int(round(start_s * source_fs))
        b = int(round(end_s * source_fs)) if end_s is not None else int(round((start_s + default_seconds) * source_fs))
        b = min(b, hdr.sig_len)
        rec = wfdb.rdrecord(base, sampfrom=a, sampto=b)
        sig = np.nan_to_num(rec.p_signal.astype(np.float64))
        names = list(leads) if leads else list(rec.sig_name)
        units = ",".join(sorted(set(u for u in rec.units if u))) if rec.units else ""
        name = os.path.basename(base)

    if sig.shape[0] < 2:
        raise ValueError("Window is empty -- check start/end seconds against the record length.")
    sig12, present, n2 = _slot_leads(sig, names)
    notes += n2
    if keep:  # emulate a reduced-lead ECG: zero every lead not listed (e.g. keep=ii,v1 ~ MIT-BIH layout)
        k = {x.strip().lower() for x in keep}
        for i, nm in enumerate(STD_LEADS):
            if nm not in k:
                sig12[:, i], present[i] = 0.0, False
        if not any(present):
            raise ValueError(f"keep={list(keep)} matches none of the recorded leads")
        notes.append(f"kept only leads {sorted(k)}; all others zero-filled")

    n_out = int(sig12.shape[0] * MODEL_FS / source_fs)   # same rounding as the repo
    if source_fs != MODEL_FS:
        sig12 = resample(sig12, n_out, axis=0)
    seconds = sig12.shape[0] / MODEL_FS
    if seconds > MAX_SECONDS:
        keep = MAX_SECONDS * MODEL_FS
        notes.append(f"window is {seconds:.0f}s; clipped to {MAX_SECONDS}s so the ECG tokens fit in the "
                     f"repo's 512-token prompt limit")
        sig12, seconds = sig12[:keep], float(MAX_SECONDS)
    win = (start_s, start_s + seconds)

    with np.errstate(invalid="ignore", divide="ignore"):   # zero-filled leads give 0/0 -> NaN -> 0 (intended)
        model_in = ecg_transform(sig12.copy()).T.astype(np.float32)   # (12, T)
    return LoadedECG(name=name, model_input=model_in, display=sig12.T, present=present,
                     source_fs=source_fs, window_s=win, units=units, notes=notes)


# --------------------------------------------------------------------------------------
# Output parsing
# --------------------------------------------------------------------------------------

_INTERVAL_RE = re.compile(r"(\d+(?:\.\d+)?)\s*s\s*-\s*(\d+(?:\.\d+)?)\s*s")


def parse_intervals(answer: str) -> List[Tuple[float, float]]:
    """'Duration: 1.9s-3.1s, 6.8s-8.1s' -> [(1.9, 3.1), (6.8, 8.1)]; 'No X Found' -> []."""
    return [(float(a), float(b)) for a, b in _INTERVAL_RE.findall(answer)]


def parse_report(answer: str) -> List[str]:
    body = answer.strip()
    if body.lower().startswith("report:"):
        body = body[len("report:"):]
    return [s.strip() for s in body.split(",") if s.strip()]


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------

def load_model(cfg: Config):
    """Build ECG_Language_Model WITHOUT touching the repo, then load the three Stage-3 pieces."""
    import torch
    from torch import nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from anyecg.ecg_language_modeling import ECG_Language_Model
    from anyecg.ecgvit import ecg_vit_base

    if not torch.cuda.is_available():
        sys.exit("[error] No CUDA GPU visible. anyECG-chat's code calls .cuda() internally, and the 8B LLM "
                 "needs ~16 GB. Run on a GPU node (or use --no-model to test ECG loading/plotting only).")

    class PlaygroundModel(ECG_Language_Model):
        """Same as the repo class but with the two author-machine paths replaced by .env values."""

        def __init__(self, ecg_encoder_ckpt: str, llm_id: str, hf_token: Optional[str]):
            # plain strings can be set before nn.Module.__init__; the parent __init__ calls the
            # two get_* methods below, which need them.
            self._pg_encoder_ckpt, self._pg_llm_id, self._pg_token = ecg_encoder_ckpt, llm_id, hf_token
            super().__init__(unfreeze_ecg_model=False, use_lora=False)

        def get_ecg_encoder(self, unfreeze_ecg_model):
            enc = ecg_vit_base()
            enc.fc = nn.Linear(enc.fc.in_features, enc.fc.out_features, bias=False)
            ckpt = torch.load(self._pg_encoder_ckpt, map_location="cpu")
            sd = {k.replace("ecg_model.model.", ""): v for k, v in ckpt.items() if "ecg_model" in k}
            res = enc.load_state_dict(sd, strict=False)
            print(f"[load] ECG encoder init from {os.path.basename(self._pg_encoder_ckpt)}: "
                  f"{len(sd)} tensors, missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)}")
            for prm in enc.parameters():
                prm.requires_grad = False
            return enc

        def get_language_model(self, use_lora):
            tok = AutoTokenizer.from_pretrained(self._pg_llm_id, token=self._pg_token)
            tok.padding_side = "left"
            tok.pad_token = tok.eos_token
            lm = AutoModelForCausalLM.from_pretrained(
                self._pg_llm_id, torch_dtype=torch.bfloat16, device_map="auto", token=self._pg_token)
            for prm in lm.parameters():
                prm.requires_grad = False
            return lm, tok

        @torch.no_grad()
        def respond(self, ecg_tensors, question: str, temperature: float, max_new_tokens: int):
            """ecg_chat() from the repo, but with greedy decoding when temperature == 0."""
            messages = [[{"role": "user", "content": question}]]   # fresh: the repo mutates it in place
            embeds, mask, _ = self.get_input_embeds(ecg_tensors, messages, add_generation_prompt=True)
            n_in = embeds.shape[1]
            if n_in >= PROMPT_LIMIT:
                print(f"[warn] prompt hit the {PROMPT_LIMIT}-token truncation limit; the answer is unreliable. "
                      f"Use a shorter ECG window or question.")
            kw = dict(inputs_embeds=embeds, attention_mask=mask, max_new_tokens=max_new_tokens,
                      eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                      pad_token_id=self.tokenizer.eos_token_id)
            if temperature > 0:
                kw.update(do_sample=True, temperature=temperature, top_p=0.9)
            else:
                kw.update(do_sample=False)
            out = self.language_model.generate(**kw)
            return self.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip(), n_in

    t0 = time.time()
    print(f"[load] LLM: {cfg.llm_id}  (first run downloads ~16 GB if this is a HF id)")
    model = PlaygroundModel(cfg.ecg_encoder_ckpt, cfg.llm_id, cfg.hf_token)

    print(f"[load] projection <- {cfg.projection_ckpt}")
    print("       ", model.projection.load_state_dict(torch.load(cfg.projection_ckpt, map_location="cpu")))
    print(f"[load] ECG encoder (stage-3 fine-tuned) <- {cfg.ecg_model_ckpt}")
    print("       ", model.ecg_model.load_state_dict(torch.load(cfg.ecg_model_ckpt, map_location="cpu")))
    try:
        model.language_model.load_adapter(cfg.lora_ckpt)
    except ValueError as e:                      # e.g. an adapter called "default" already exists
        print(f"[load] load_adapter default name failed ({e}); retrying as 'anyecg_stage3'")
        model.language_model.load_adapter(cfg.lora_ckpt, adapter_name="anyecg_stage3")
    print(f"[load] LoRA adapter <- {cfg.lora_ckpt}   active: {model.language_model.active_adapters()}")

    # LLM is already placed by device_map="auto"; move the small trainable-size parts to the GPU.
    model.ecg_model.cuda()
    model.projection.cuda()
    model.eval()
    print(f"[load] ready in {time.time() - t0:.0f}s")
    warnings.filterwarnings("ignore", message="To copy construct from a tensor")
    return model


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------

def plot_ecg(ecg: LoadedECG, answers: Optional[List[List[Tuple[float, float]]]] = None,
             title: str = "", out_path: Optional[Path] = None) -> Optional[Path]:
    """ECG traces (recorded leads only). `answers` = one interval list per sampled model answer."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = [i for i, ok in enumerate(ecg.present) if ok]
    t = np.arange(ecg.display.shape[1]) / MODEL_FS
    fig, axes = plt.subplots(len(idx), 1, sharex=True, figsize=(14, max(2.2 * len(idx), 3.2)), squeeze=False)
    cover = None
    if answers:
        cover = np.zeros_like(t)
        for ivs in answers:
            m = np.zeros_like(t)
            for a, b in ivs:
                m[(t >= a) & (t <= b)] = 1.0
            cover += m
        cover /= len(answers)
    for ax, i in zip(axes[:, 0], idx):
        ax.plot(t, ecg.display[i], lw=0.8, color="black")
        ax.set_ylabel(STD_LEADS[i].upper() + (f" ({ecg.units})" if ecg.units and ax is axes[0, 0] else ""))
        ax.margins(x=0)
        if cover is not None and cover.max() > 0:
            ymin, ymax = ax.get_ylim()
            # draw coverage as a strip of constant-alpha spans so alpha == fraction of answers covering
            edges = np.flatnonzero(np.diff(cover) != 0) + 1
            starts = np.r_[0, edges]
            ends = np.r_[edges, len(t)]
            for s, e in zip(starts, ends):
                if cover[s] > 0:
                    ax.axvspan(t[s], t[min(e, len(t) - 1)], color="red", alpha=0.08 + 0.32 * cover[s], lw=0)
            ax.set_ylim(ymin, ymax)
    axes[-1, 0].set_xlabel(f"seconds from start of window (source recording {ecg.window_s[0]:.1f}s-{ecg.window_s[1]:.1f}s)")
    fig.suptitle(title or ecg.name, fontsize=11)
    fig.tight_layout()
    if out_path is None:
        plt.close(fig)
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------------
# Session
# --------------------------------------------------------------------------------------

class Session:
    def __init__(self, cfg: Config, model, args):
        self.cfg, self.model = cfg, model
        self.ecg: Optional[LoadedECG] = None
        self.temperature = args.temperature if args.temperature is not None else cfg.temperature
        self.samples = args.samples
        self.max_new_tokens = cfg.max_new_tokens
        self.seed: Optional[int] = args.seed
        self.out_dir = Path(args.output_dir or cfg.output_dir)
        self.default_seconds = args.duration
        self.log_path = self.out_dir / "results.jsonl"

    # -- plumbing -------------------------------------------------------------------
    def _need_ecg(self) -> LoadedECG:
        if self.ecg is None:
            raise RuntimeError("No ECG loaded. Use: load <record> [start_s [end_s]]")
        return self.ecg

    def _need_model(self):
        if self.model is None:
            raise RuntimeError("Started with --no-model: only load/plot/info are available.")
        return self.model

    def _log(self, kind: str, **fields):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        rec = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), kind=kind, ecg=self.ecg.name if self.ecg else None,
                   window_s=self.ecg.window_s if self.ecg else None, temperature=self.temperature, **fields)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def _generate(self, question: str) -> Tuple[str, int]:
        import torch
        ecg = self._need_ecg()
        m = self._need_model()
        if self.seed is not None:
            torch.manual_seed(self.seed)
        tens = [torch.from_numpy(ecg.model_input).float()]   # list of (12, T) tensors, as the repo's collate_fn gives
        return m.respond(tens, question, self.temperature, self.max_new_tokens)

    # -- commands -------------------------------------------------------------------
    def load(self, spec: str, start: Optional[float] = None, end: Optional[float] = None,
             fs: Optional[float] = None, leads: Optional[Sequence[str]] = None,
             keep: Optional[Sequence[str]] = None):
        self.ecg = load_ecg(spec, start, end, self.cfg.ecg_dir, fs=fs, leads=leads,
                            default_seconds=self.default_seconds, keep=keep)
        self.info()
        for n in self.ecg.notes:
            print(f"  [note] {n}")

    def info(self):
        e = self._need_ecg()
        n_clips = math.ceil(e.model_input.shape[1] / CLIP_SAMPLES)
        have = [STD_LEADS[i].upper() for i, ok in enumerate(e.present) if ok]
        print(f"  {e.name}: {e.seconds:.1f}s window ({e.window_s[0]:.1f}s-{e.window_s[1]:.1f}s of source @ "
              f"{e.source_fs:g} Hz -> {MODEL_FS} Hz), leads: {' '.join(have)} "
              f"(+{12 - len(have)} zero-filled), {n_clips} x 10s clip(s) = {1 + TOKENS_PER_CLIP * n_clips} ECG tokens")

    def report(self):
        t0 = time.time()
        ans, n_in = self._generate(REPORT_QUESTION)
        print(f"\n  {ans}\n")
        is_report = ans.lower().startswith("report:")
        findings = parse_report(ans) if is_report else []
        for i, f in enumerate(findings, 1):
            print(f"   {i:>2}. {f}")
        print(f"  [{time.time() - t0:.1f}s, prompt {n_in} tokens, T={self.temperature}]")
        if not is_report:
            n_leads = sum(self._need_ecg().present)
            print("  [warn] answer is not in 'Report: ...' format, so it is not a report. "
                  + (f"Only {n_leads} of 12 leads are recorded here; the report task was trained on 12-lead MIMIC ECGs, "
                     "while 2-lead MIT-BIH/ST-T ECGs were only seen in the localisation task. " if n_leads < 12 else "")
                  + "Try a 12-lead ECG, `lora off`, `set temperature 0`, or re-run.")
        self._log("report", question=REPORT_QUESTION, answer=ans, findings=findings, is_report_format=is_report)

    def locate(self, name: str):
        e = self._need_ecg()
        name = name.strip().strip('"\'')   # tolerate: locate "atrial fibrillation"
        phrase, in_vocab = resolve_abnormality(name)
        if not in_vocab:
            print(f"  [warn] '{phrase}' is not one of the trained localisation targets "
                  f"({', '.join(ABNORMALITIES)}); expect unreliable output.")
        question = LOCATE_TEMPLATE.replace("{X}", phrase)
        k = max(1, self.samples)
        if k > 1 and self.temperature <= 0:
            print("  [warn] --samples > 1 with temperature 0 gives identical answers; use `set temperature 0.6`.")
        answers, ivs_all = [], []
        t0 = time.time()
        for _ in range(k):
            ans, _ = self._generate(question)
            answers.append(ans)
            ivs_all.append(parse_intervals(ans))
        print(f"\n  Q: {question}")
        for i, (a, ivs) in enumerate(zip(answers, ivs_all), 1):
            tag = f"[{i}/{k}] " if k > 1 else ""
            print(f"  A {tag}{a}")
        print(f"  [{time.time() - t0:.1f}s, T={self.temperature}]")
        found = sum(1 for v in ivs_all if v)
        odd = [a for a, v in zip(answers, ivs_all) if not v and "found" not in a.lower()]
        if odd:
            print(f"  [warn] {len(odd)}/{k} answer(s) are neither a 'Duration:' span nor 'No ... Found' "
                  f"(e.g. {odd[0]!r}): the model did not treat this as a localisation request.")
        if k > 1:
            print(f"  {found}/{k} answers contained at least one interval")
        # intervals are relative to window start; clip to window
        clipped = [[(max(0.0, a), min(e.seconds, b)) for a, b in v if b > 0 and a < e.seconds] for v in ivs_all]
        if any(a > e.seconds for v in ivs_all for a, _ in v):
            print("  [warn] some answered times lie beyond the ECG window (model hallucinated a span).")
        stamp = time.strftime("%H%M%S")
        png = plot_ecg(e, clipped if found else None,
                       title=f"{e.name} | {phrase} | "
                             + (f"agreement over {k} samples" if k > 1 else (answers[0][:80])),
                       out_path=self.out_dir / f"{e.name}_locate_{re.sub(r'[^a-z0-9]+', '', phrase.lower())[:20]}_{stamp}.png")
        print(f"  plot: {png}")
        self._log("locate", abnormality=phrase, question=question, answers=answers, intervals=ivs_all,
                  plot=str(png))

    def ask(self, question: str):
        ans, n_in = self._generate(question)
        print(f"\n  {ans}\n  [prompt {n_in} tokens, T={self.temperature}]")
        self._log("ask", question=question, answer=ans)

    def plot(self):
        e = self._need_ecg()
        png = plot_ecg(e, title=f"{e.name} ({e.window_s[0]:.1f}s-{e.window_s[1]:.1f}s)",
                       out_path=self.out_dir / f"{e.name}_raw_{time.strftime('%H%M%S')}.png")
        print(f"  plot: {png}")

    def lora(self, state: str):
        lm = self._need_model().language_model
        if state == "off":
            lm.disable_adapters()
        elif state == "on":
            lm.enable_adapters()
        else:
            raise ValueError("usage: lora on|off")
        print(f"  LoRA adapter {state}.")

    def set(self, key: str, val: str):
        if key == "temperature":
            self.temperature = float(val)
        elif key == "samples":
            self.samples = max(1, int(val))
        elif key == "max_new_tokens":
            self.max_new_tokens = int(val)
        elif key == "seed":
            self.seed = None if val.lower() in ("none", "off") else int(val)
        elif key == "duration":
            self.default_seconds = float(val)
        else:
            raise ValueError("settable: temperature, samples, max_new_tokens, seed, duration")
        print(f"  {key} = {val}")

    def settings(self):
        print(f"  temperature={self.temperature} samples={self.samples} max_new_tokens={self.max_new_tokens} "
              f"seed={self.seed} default_duration={self.default_seconds}s outputs={self.out_dir}")


HELP = """\
  load <record> [start_s [end_s]] [fs=500] [leads=i,ii] [keep=ii,v1]
        keep=: zero all other leads (mimics the 2-lead training layout on a 12-lead ECG)
        WFDB record base path (no extension) or .npy/.csv (needs fs=). Relative names resolve against ECG_DIR.
        Default window = first --duration seconds (10). Up to 70 s; >10 s is processed as 10 s clips.
  report                       "Please provide the report for the following ECG."
  locate <name>                pvc | lbbb | rbbb | pac | svpb | free text ("Where is the <name> located...")
  ask <question>               free-form question about the loaded ECG
  set temperature|samples|max_new_tokens|seed|duration <v>
        temperature 0 = greedy. samples K>1 on `locate` = K answers + agreement shading.
  lora on|off                  disable/enable the adapter (answers should change if it is really applied)
  plot | info | settings | help | quit
"""


def repl(sess: Session):
    try:
        import readline  # noqa: F401  (history/arrow keys)
    except ImportError:
        pass
    print("\nType `help` for commands. Example:  load /data/mitdb/100 0 10   then   locate pvc\n")
    while True:
        try:
            line = input("anyecg> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        cmd, _, rest = line.partition(" ")
        cmd, rest = cmd.lower(), rest.strip()
        try:
            if cmd in ("quit", "exit", "q"):
                return
            elif cmd == "help":
                print(HELP)
            elif cmd == "load":
                toks = rest.split()
                if not toks:
                    raise ValueError("usage: load <record> [start_s [end_s]]")
                kv = {t.split("=", 1)[0]: t.split("=", 1)[1] for t in toks[1:] if "=" in t}
                pos = [t for t in toks[1:] if "=" not in t]
                sess.load(toks[0], float(pos[0]) if len(pos) > 0 else None, float(pos[1]) if len(pos) > 1 else None,
                          fs=float(kv["fs"]) if "fs" in kv else None,
                          leads=kv["leads"].split(",") if "leads" in kv else None,
                          keep=kv["keep"].split(",") if "keep" in kv else None)
            elif cmd == "report":
                sess.report()
            elif cmd == "locate":
                if not rest:
                    raise ValueError("usage: locate <pvc|lbbb|rbbb|pac|svpb|free text>")
                sess.locate(rest)
            elif cmd == "ask":
                if not rest:
                    raise ValueError("usage: ask <question>")
                sess.ask(rest)
            elif cmd == "set":
                k, _, v = rest.partition(" ")
                sess.set(k, v.strip())
            elif cmd == "lora":
                sess.lora(rest.lower())
            elif cmd == "plot":
                sess.plot()
            elif cmd == "info":
                sess.info()
            elif cmd == "settings":
                sess.settings()
            else:
                print(f"  unknown command '{cmd}' (try `help`)")
        except Exception as e:  # keep the session alive on any error
            print(f"  [error] {type(e).__name__}: {e}")


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Interactive sandbox for anyECG-chat report generation + localisation.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Config comes from .env -- see .env.example.")
    ap.add_argument("--env", help="path to a .env file (default: ./.env then <script dir>/.env)")
    ap.add_argument("--ecg", help="ECG to load at start (WFDB record base path, .npy or .csv)")
    ap.add_argument("--start", type=float, help="window start in seconds (default 0)")
    ap.add_argument("--end", type=float, help="window end in seconds (default start + --duration)")
    ap.add_argument("--duration", type=float, default=10.0, help="default window length in seconds (default 10)")
    ap.add_argument("--fs", type=float, help="sampling rate for .npy/.csv input")
    ap.add_argument("--leads", help="comma-separated lead names for each channel (override auto-detection)")
    ap.add_argument("--keep", help="keep only these leads, zero the rest (e.g. ii,v1 to mimic 2-lead training data)")
    ap.add_argument("--report", action="store_true", help="generate a report for --ecg, then exit unless -i")
    ap.add_argument("--locate", metavar="NAME", help="localise NAME (pvc/lbbb/rbbb/pac/svpb/free text) in --ecg")
    ap.add_argument("--ask", metavar="QUESTION", help="ask a free-form question about --ecg")
    ap.add_argument("-i", "--interactive", action="store_true", help="drop into the prompt after one-shot actions")
    ap.add_argument("--temperature", type=float, help="sampling temperature (0 = greedy); default 0.6 / $TEMPERATURE")
    ap.add_argument("--samples", type=int, default=1, help="answers to sample per `locate` (agreement shading)")
    ap.add_argument("--seed", type=int, help="torch seed set before every generation")
    ap.add_argument("--output-dir", help="where plots + results.jsonl go (default $OUTPUT_DIR or ./anyecg_outputs)")
    ap.add_argument("--check", action="store_true", help="validate .env and imports, then exit (no model load)")
    ap.add_argument("--no-model", action="store_true", help="skip loading the model (test ECG loading/plotting)")
    args = ap.parse_args()

    env_file = load_env(args.env)
    print(f"[env] {env_file if env_file else 'no .env found (using shell environment only)'}")
    cfg = build_config(need_model=not args.no_model)

    sys.path.append(cfg.repo)            # makes `import anyecg...` work without touching the repo
    if cfg.hf_token:
        os.environ.setdefault("HF_TOKEN", cfg.hf_token)
    if os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = os.path.expanduser(os.environ["HF_HOME"])
    print(f"[env] repo: {cfg.repo}")

    if args.check:
        import importlib
        bad = []
        for mod in ("torch", "transformers", "peft", "einops", "wfdb", "scipy", "numpy", "matplotlib",
                    "anyecg.ecgvit", "anyecg.ecg_language_modeling", "anyecg.utils"):
            try:
                importlib.import_module(mod)
            except Exception as e:  # noqa: BLE001
                bad.append(f"{mod}: {type(e).__name__}: {e}")
        for line in bad:
            print("  [missing]", line)
        try:
            import torch
            print(f"[check] CUDA available: {torch.cuda.is_available()}")
        except Exception:
            pass
        print(f"[check] llm={cfg.llm_id}\n[check] lora={cfg.lora_ckpt}\n[check] projection={cfg.projection_ckpt}\n"
              f"[check] ecg_model={cfg.ecg_model_ckpt}\n[check] ecg_encoder={cfg.ecg_encoder_ckpt}")
        sys.exit(1 if bad else 0)

    model = None if args.no_model else load_model(cfg)
    sess = Session(cfg, model, args)

    one_shot = bool(args.report or args.locate or args.ask)
    if args.ecg:
        try:
            sess.load(args.ecg, args.start, args.end, fs=args.fs,
                      leads=args.leads.split(",") if args.leads else None,
                      keep=args.keep.split(",") if args.keep else None)
        except Exception as e:  # noqa: BLE001
            sys.exit(f"[error] could not load ECG: {e}")
        if args.no_model:
            sess.plot()
    elif one_shot:
        sys.exit("[error] --report/--locate/--ask need --ecg")

    if one_shot:
        if args.report:
            sess.report()
        if args.locate:
            sess.locate(args.locate)
        if args.ask:
            sess.ask(args.ask)
    if not one_shot or args.interactive:
        repl(sess)


if __name__ == "__main__":
    main()