"""Segmentation-finetune task adapter (flywheel leg 3): a DatasetManifest in, a
workspace-local training run out — weights + TrainingRunManifest. Capability-
INTERNAL per DEC d02a38d4 (the generic finetune-adapter interface lib is minted
only at n=2 trainable capabilities); the manifest chain closes per DECs
16159e09 + e047beee: source -> decomp manifest -> correction journal ->
dataset manifest -> THIS -> model."""

import argparse
import hashlib
import json
import logging
import os
import platform
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List, Optional, Tuple, Union

from cjm_substrate.core.capability import ToolCapability
from cjm_substrate.core.errors import CapabilityFatalError, CapabilityInputError
from cjm_substrate.core.workspace import (relativize_recorded, resolve_recorded_tree,
                                          resolve_workspace, Workspace)
from cjm_substrate.utils.validation import (config_to_dict, dataclass_to_jsonschema, dict_to_config,
                                            SCHEMA_DESC, SCHEMA_TITLE)

# The adapter wraps the standard pyannote finetune recipe: warm-start
# pyannote/segmentation-3.0 (HF-GATED — same env-provisioning contract as the
# diarization pipeline: hf auth login or HF_TOKEN, never a config knob) with a
# MultiLabelSegmentation task whose classes come from the dataset's observed
# OPEN vocabulary. Training runs ride the worker regime that carries long
# inference runs — no job-queue machinery (DEC d02a38d4).

# Training Imports (heavy; the adapter reports unavailable when missing)
try:
    import scipy.io.wavfile
    import torch
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.loggers import CSVLogger
    from pyannote.audio import Model
    from pyannote.audio.core.io import Audio
    from pyannote.audio.tasks import MultiLabelSegmentation
    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.database.protocol import SegmentationProtocol
    TRAIN_AVAILABLE = True
except ImportError:
    TRAIN_AVAILABLE = False
    SegmentationProtocol = object  # class-def base fallback so the module imports without the stack

# Format tag of the consumed dataset manifest (born workflow-local in
# cjm-transcript-correction-core; consumed BY POINTER, DEC 16159e09)
DATASET_MANIFEST_FORMAT = "cjm-transcript-correction-core/dataset-manifest"

# Workspace-local landing zone for training runs (a training run is a run —
# outputs stay workspace-local; promotion to the system model cache happens on
# graduation, DEC e047beee)
TRAINING_RUNS_DIR = "training-runs"

# Shared per-source 16 kHz mono wav cache under the training-runs dir, keyed by
# source content hash — one transcode per source, reused across runs
AUDIO_CACHE_DIR = "_audio_cache"

# Auxiliary dense class derived from the dataset's speech regions: keeps the
# base model's speech competence anchored while the sparse event classes train
SPEECH_CLASS = "speech"

# Model input sample rate (segmentation-3.0 contract)
SAMPLE_RATE = 16000


def _hf_token() -> Optional[str]:
    """The HF token from the worker env (gated-model pull); None falls back to
    the hub's cached login (hf auth login) — provisioning, not config."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def new_training_run_id() -> str:  # e.g. "trainrun_20260729_170000_1a2b3c4d"
    """Generate a unique, sortable training-run id (the new_run_id pattern, trainrun kind)."""
    return f"trainrun_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


@dataclass
class TrainingRunManifest:
    """Durable record of one finetune run (chainable; DEC 16159e09).

    Owned by the training capability lib: format
    `cjm-capability-pyannote/training-run-manifest`. Extends the manifest
    pattern (format tag + version + consumed pointer + WS-token paths):
    consumes a DatasetManifest BY POINTER, records base model identity +
    revision, the hyperparameter snapshot, capability lib + env versions, the
    output artifact + content hash, and REAL-ONLY holdout eval results
    (trust-ladder discipline, DEC 03d207cf). The manifest IS the registration
    record — identity = run id + content hash, recorded not path-encoded
    (DEC e047beee); promotion to the system model cache copies the run
    directory, manifest included."""
    run_id: str              # Unique training-run identifier (sortable, run-id pattern)
    created_at: float        # Unix timestamp at run start
    config: Dict[str, Any]   # Hyperparameter snapshot (FinetuneConfig as dict)
    dataset_manifest: str    # Recorded path of the consumed DatasetManifest (the pointer)
    dataset_id: str          # Consumed dataset id (join key for manifest-chain queries)
    base_model: Dict[str, Any] = field(default_factory=dict)   # Base model identity: model_id + hub revision
    environment: Dict[str, Any] = field(default_factory=dict)  # Capability lib + version + runtime env versions
    classes: List[str] = field(default_factory=list)           # Trained class list (model output order)
    files: Dict[str, str] = field(default_factory=dict)        # Run-relative artifact files (checkpoint/logs)
    artifact: Dict[str, Any] = field(default_factory=dict)     # Output checkpoint: recorded path + sha256 + size
    eval: Dict[str, Any] = field(default_factory=dict)         # Real-only holdout policy + metrics
    counts: Dict[str, Any] = field(default_factory=dict)       # Per-spine + total train/dev example counts

    FORMAT: str = field(default="cjm-capability-pyannote/training-run-manifest", repr=False)  # Format tag
    VERSION: str = field(default="0.1.0", repr=False)                                          # Schema version

    def to_dict(self) -> Dict[str, Any]:  # Plain-dict form for JSON serialization
        """Serialize to a plain dict."""
        return {
            "format": self.FORMAT,
            "version": self.VERSION,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "config": self.config,
            "dataset_manifest": self.dataset_manifest,
            "dataset_id": self.dataset_id,
            "base_model": self.base_model,
            "environment": self.environment,
            "classes": list(self.classes),
            "files": dict(self.files),
            "artifact": self.artifact,
            "eval": self.eval,
            "counts": self.counts,
        }

    def save(
        self,
        path: Union[str, Path],  # Destination JSON file (parent dirs created)
        workspace=None,  # Active Workspace; owned paths record as ${WS}/<rel> (5daadfc4 rung f)
    ) -> Path:  # The written path
        """Write the manifest as pretty-printed JSON (WS-token recorded paths —
        the run directory relocates with the workspace, DEC a5883992)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(relativize_recorded(self.to_dict(), workspace), indent=2))
        return out


@dataclass
class FinetuneConfig:
    """Configuration for one segmentation finetune run.

    Every knob lands verbatim in the TrainingRunManifest's config snapshot —
    the manifest, not the directory layout, is what makes a run reproducible."""

    base_model_id: str = field(
        default="pyannote/segmentation-3.0",
        metadata={
            SCHEMA_TITLE: "Base Model ID",
            SCHEMA_DESC: "HuggingFace model id to warm-start from (gated: accept the "
                         "model's user conditions and provision a token in the worker env).",
        }
    )
    device: str = field(
        default="auto",
        metadata={
            SCHEMA_TITLE: "Device",
            SCHEMA_DESC: "Training device: 'auto' (cuda when available), 'cuda', 'cuda:N', or 'cpu'.",
        }
    )
    duration: float = field(
        default=10.0,
        metadata={
            SCHEMA_TITLE: "Chunk Duration",
            SCHEMA_DESC: "Training chunk length in seconds (segmentation-3.0 was trained on 10s chunks).",
        }
    )
    batch_size: int = field(
        default=32,
        metadata={SCHEMA_TITLE: "Batch Size", SCHEMA_DESC: "Training chunks per batch."}
    )
    max_epochs: int = field(
        default=20,
        metadata={SCHEMA_TITLE: "Max Epochs", SCHEMA_DESC: "Number of training epochs."}
    )
    learning_rate: float = field(
        default=1e-3,
        metadata={SCHEMA_TITLE: "Learning Rate", SCHEMA_DESC: "Adam learning rate (the recipe default)."}
    )
    num_workers: int = field(
        default=4,
        metadata={SCHEMA_TITLE: "Dataloader Workers", SCHEMA_DESC: "Chunk-generation worker processes."}
    )
    holdout_fraction: float = field(
        default=0.1,
        metadata={
            SCHEMA_TITLE: "Holdout Fraction",
            SCHEMA_DESC: "Time-tail fraction of each spine's ANNOTATED HEAD held out for "
                         "real-only eval (the reserved tail above the watermark is the live "
                         "bench and is never read here).",
        }
    )
    min_class_count: int = field(
        default=5,
        metadata={
            SCHEMA_TITLE: "Min Class Count",
            SCHEMA_DESC: "Observed vocabulary labels with fewer examples than this are not "
                         "trained as classes (their spans stay annotated negatives).",
        }
    )
    exclude_labels: List[str] = field(
        default_factory=lambda: ["empty"],
        metadata={
            SCHEMA_TITLE: "Excluded Labels",
            SCHEMA_DESC: "Labels never trained as classes. 'empty' is the hard-negative "
                         "insert class — its spans are evidence of nothing, not an event.",
        }
    )
    include_speech_class: bool = field(
        default=True,
        metadata={
            SCHEMA_TITLE: "Speech Class",
            SCHEMA_DESC: "Add an auxiliary dense 'speech' class from the dataset's speech "
                         "regions (anchors the base model's speech competence).",
        }
    )
    prepare_wav: bool = field(
        default=True,
        metadata={
            SCHEMA_TITLE: "Prepare WAV",
            SCHEMA_DESC: "Transcode each source once to a cached 16 kHz mono wav (exact "
                         "random access; avoids per-chunk compressed-audio seeks).",
        }
    )
    seed: int = field(
        default=42,
        metadata={SCHEMA_TITLE: "Seed", SCHEMA_DESC: "Global RNG seed (lightning seed_everything)."}
    )


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a jsonl file into a list of dicts."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_dataset(
    manifest_path: Union[str, Path]  # Path to a DatasetManifest json (the consumed pointer)
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:  # (resolved manifest, events, regions)
    """Load a DatasetManifest and its data files.

    Recorded ${WS} paths resolve via resolve_recorded_tree (reader half of the
    recording contract); the events/regions files are dataset-dir-relative."""
    p = Path(manifest_path).resolve()
    data = json.loads(p.read_text())
    fmt = data.get("format")
    if fmt != DATASET_MANIFEST_FORMAT:
        raise CapabilityInputError(
            f"{p} is not a dataset manifest (format {fmt!r}; expected {DATASET_MANIFEST_FORMAT!r})",
            fields_invalid=["dataset_manifest"],
        )
    resolved = resolve_recorded_tree(data, p)
    events = _load_jsonl(p.parent / resolved["files"]["events"])
    regions = _load_jsonl(p.parent / resolved["files"]["regions"])
    return resolved, events, regions


def select_classes(
    manifest: Dict[str, Any],  # Resolved dataset manifest
    cfg: FinetuneConfig        # Active finetune config
) -> List[str]:  # Trained class list (event classes by count desc, speech last)
    """Pick the trained classes from the dataset's observed OPEN vocabulary.

    Labels below min_class_count or in exclude_labels are NOT trained — their
    spans remain inside the annotated timeline, so they count as negative
    evidence rather than becoming starved output heads."""
    vocab: Dict[str, int] = manifest.get("class_vocabulary", {})
    excluded = set(cfg.exclude_labels)
    kept = sorted(
        ((count, label) for label, count in vocab.items()
         if count >= cfg.min_class_count and label not in excluded),
        key=lambda pair: (-pair[0], pair[1]),
    )
    classes = [label for _count, label in kept]
    if cfg.include_speech_class:
        classes.append(SPEECH_CLASS)
    if not classes:
        raise CapabilityInputError(
            f"No trainable classes: vocabulary {vocab} under min_class_count="
            f"{cfg.min_class_count} / exclude_labels={sorted(excluded)}",
            fields_invalid=["min_class_count", "exclude_labels"],
        )
    return classes


def _eligible_spines(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The gate-eligible spines recorded at extraction time (watermark holders)."""
    return [s for s in manifest.get("spines", []) if s.get("eligible")]


def _prepare_wav(
    source_path: str,             # Original source audio (any format/rate)
    content_hash: Optional[str],  # Source content hash from the dataset (cache key)
    cache_dir: Path,              # Shared audio cache dir (created on demand)
    logger: logging.Logger        # Run logger
) -> Path:  # Cached 16 kHz mono float32 wav (coordinates unchanged)
    """Transcode a source once into the shared 16 kHz mono wav cache.

    A full-file decode keeps the time axis identical to the annotation
    coordinates; the wav then gives the chunk sampler exact random access."""
    key = (content_hash or f"sha256:{hashlib.sha256(source_path.encode()).hexdigest()}")
    key = key.split(":", 1)[-1][:16]
    out = cache_dir / f"{key}_{SAMPLE_RATE}hz.wav"
    if out.exists():
        return out
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Transcoding {source_path} -> {out}")
    waveform, _sr = Audio(sample_rate=SAMPLE_RATE, mono="downmix")({"audio": source_path})
    tmp = out.with_suffix(".tmp.wav")
    scipy.io.wavfile.write(tmp, SAMPLE_RATE, waveform.squeeze(0).numpy())
    tmp.rename(out)  # atomic publish — a crashed transcode never poisons the cache
    return out


def _build_spine_files(
    spine: Dict[str, Any],           # Eligible spine entry (gate state at extraction)
    events: List[Dict[str, Any]],    # This spine's labeled spans
    regions: List[Dict[str, Any]],   # This spine's speech/negative regions
    classes: List[str],              # Trained class list
    cfg: FinetuneConfig,             # Active finetune config
    audio_path: Path                 # Audio the sampler reads (original or cached wav)
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:  # (train file, dev file, counts)
    """Build the train/dev protocol files for one spine.

    The annotated head [0, watermark] splits by TIME: head minus the holdout
    tail trains, the holdout tail is the real-only dev set. Everything above
    the watermark (the reserved tail) was never extracted and never appears
    here — it is the live bench (DEC 8cf12c22). Within an annotated window,
    absence of a class is a TRUE negative: the leg-2 fold guarantees full
    annotation below the watermark."""
    watermark = float(spine["annotated_through"])
    dev_start = watermark * (1.0 - cfg.holdout_fraction)
    uri_base = f"{spine['source_id'][:8]}-{spine['skeleton_hash'].split(':', 1)[-1][:8]}"
    class_set = set(classes)

    skipped = sum(spine.get("skipped", {}).values())
    if skipped:
        logging.getLogger(__name__).warning(
            f"Spine {uri_base}: {skipped} spans were skipped at extraction; their regions "
            "are treated as negatives here (v1 limitation — the dataset does not itemize them)."
        )

    def build_annotation(uri: str, window: "Segment", real_only: bool) -> "Annotation":
        ann = Annotation(uri=uri)
        track = 0
        for ev in events:
            if ev["label"] not in class_set:
                continue
            if real_only and ev.get("provenance", {}).get("tag", "real") != "real":
                continue
            seg = Segment(ev["start_time"], ev["end_time"]) & window
            if seg and seg.duration > 0:
                ann[seg, track] = ev["label"]
                track += 1
        if SPEECH_CLASS in class_set:
            for region in regions:
                if region.get("kind") != "speech":
                    continue
                seg = Segment(region["start_time"], region["end_time"]) & window
                if seg and seg.duration > 0:
                    ann[seg, track] = SPEECH_CLASS
                    track += 1
        return ann

    def build_file(subset: str, window: "Segment", real_only: bool) -> Dict[str, Any]:
        uri = f"{uri_base}-{subset}"
        return {
            "uri": uri,
            "database": "cjm-flywheel",
            # MUST be "global": MultiLabelSegmentation targets index by
            # global_label_idx, which prepare_data populates only at global
            # scope — any narrower scope leaves -1, silently training every
            # span into the LAST class column (caught live 2026-07-29: inhale
            # head trained all-zero, tail probabilities ~1e-5)
            "scope": "global",
            "audio": str(audio_path),
            "annotated": Timeline([window], uri=uri),
            "annotation": build_annotation(uri, window, real_only),
            "classes": list(classes),
        }

    train_window = Segment(0.0, dev_start)
    dev_window = Segment(dev_start, watermark)
    train_file = build_file("train", train_window, real_only=False)
    dev_file = build_file("dev", dev_window, real_only=True)

    def label_counts(ann: "Annotation") -> Dict[str, int]:
        out: Dict[str, int] = {}
        for _seg, _track, label in ann.itertracks(yield_label=True):
            out[label] = out.get(label, 0) + 1
        return out

    counts = {
        "uri": uri_base,
        "watermark": watermark,
        "dev_window": [dev_start, watermark],
        "train": label_counts(train_file["annotation"]),
        "dev": label_counts(dev_file["annotation"]),
    }
    return train_file, dev_file, counts


def encounter_class_order(
    files: List[Dict[str, Any]]  # Protocol files in prepare_data iteration order (train then dev)
) -> List[str]:  # Labels by first occurrence across all annotations
    """The class order upstream target-building actually uses.

    Task.prepare_data accumulates global labels in ENCOUNTER order (first
    occurrence across files, chronological within each file), and
    MultiLabelSegmentation.prepare_chunk indexes target columns by that order
    while classes-list keeps the caller's order — there is NO remapping
    between the two. Passing classes in any other order silently permutes the
    training targets, so the task's classes MUST be exactly this list."""
    seen: List[str] = []
    for f in files:
        for _seg, _track, label in f["annotation"].itertracks(yield_label=True):
            if label not in seen:
                seen.append(label)
    return seen


class _DatasetProtocol(SegmentationProtocol):
    """In-memory SegmentationProtocol over the dataset's protocol files — no
    registry/database.yml/RTTM round-trip; file dicts carry audio/annotated/
    annotation/classes directly."""

    def __init__(self, train_files: List[Dict[str, Any]], dev_files: List[Dict[str, Any]],
                 name: str = "cjm-flywheel.Segmentation.dataset"):
        super().__init__()
        self.name = name  # Task.prepare_data records this (registry protocols carry it)
        self._train_files = train_files
        self._dev_files = dev_files

    def train_iter(self):
        yield from self._train_files

    def development_iter(self):
        yield from self._dev_files


def _trainer_devices(device: str) -> Tuple[str, Any]:  # (accelerator, devices) for the Trainer
    """Map the device knob onto lightning Trainer arguments."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return "cpu", 1
    if device.startswith("cuda:"):
        return "gpu", [int(device.split(":", 1)[1])]
    return "gpu", 1


def _resolve_revision(model_id: str) -> Optional[str]:
    """Best-effort hub revision (commit sha) of the base model; None offline."""
    try:
        from huggingface_hub import model_info
        return model_info(model_id, token=_hf_token()).sha
    except Exception:
        return None


def _environment_record() -> Dict[str, Any]:
    """Versions that shaped this run (manifest environment block)."""
    import lightning
    import pyannote.audio
    from cjm_capability_pyannote import __version__
    record = {
        "capability": "cjm-capability-pyannote",
        "capability_version": __version__,
        "python": platform.python_version(),
        "pyannote.audio": pyannote.audio.__version__,
        "torch": torch.__version__,
        "lightning": lightning.__version__,
    }
    if torch.cuda.is_available():
        record["gpu"] = torch.cuda.get_device_name(0)
    return record


def _holdout_eval(
    model,                            # The finetuned model (post-fit)
    dev_files: List[Dict[str, Any]],  # Real-only dev protocol files
    classes: List[str],               # Trained class list (model output order)
    cfg: FinetuneConfig               # Active finetune config
) -> Dict[str, Any]:  # Frame-level per-class AUROC + average precision on the holdout
    """Score the real-only holdout per class.

    Upstream MultiLabelSegmentation logs only BCE val loss — its default AUROC
    metric is never wired into validation_step (a TODO in pyannote.audio) — so
    the adapter runs its own sliding-window inference over each dev window and
    scores frame-level AUROC + average precision per class. Classes with no
    positive holdout frames record null, never a fabricated number."""
    from pyannote.audio import Inference
    from torchmetrics.functional.classification import binary_auroc, binary_average_precision
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device != "cpu" else "cpu")
    inference = Inference(model, duration=cfg.duration, step=cfg.duration / 2.0,
                          batch_size=cfg.batch_size, device=device)
    all_scores: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    for f in dev_files:
        window = next(iter(f["annotated"]))
        scores = inference.crop({"audio": f["audio"], "uri": f["uri"]}, window)
        frames = scores.sliding_window
        n = scores.data.shape[0]
        spans: Dict[str, List[Any]] = {c: [] for c in classes}
        for seg, _track, label in f["annotation"].itertracks(yield_label=True):
            spans[label].append(seg)
        target = torch.zeros((n, len(classes)), dtype=torch.long)
        for i in range(n):
            t = frames[i].middle
            for j, c in enumerate(classes):
                if any(s.start <= t < s.end for s in spans[c]):
                    target[i, j] = 1
        all_scores.append(torch.as_tensor(scores.data, dtype=torch.float32))
        all_targets.append(target)
    scores_t = torch.cat(all_scores)
    targets_t = torch.cat(all_targets)
    out: Dict[str, Any] = {"frames": int(scores_t.shape[0]), "positive_frames": {},
                           "auroc": {}, "average_precision": {}}
    for j, c in enumerate(classes):
        pos = int(targets_t[:, j].sum())
        out["positive_frames"][c] = pos
        if 0 < pos < targets_t.shape[0]:
            out["auroc"][c] = float(binary_auroc(scores_t[:, j], targets_t[:, j]))
            out["average_precision"][c] = float(binary_average_precision(scores_t[:, j], targets_t[:, j]))
        else:
            out["auroc"][c] = None
            out["average_precision"][c] = None
    for key in ("auroc", "average_precision"):
        defined = [v for v in out[key].values() if v is not None]
        out[key]["macro"] = float(sum(defined) / len(defined)) if defined else None
    return out


def _file_sha256(path: Path) -> str:
    """Streaming sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_finetune(
    dataset_manifest: Union[str, Path],       # DatasetManifest json path (the consumed pointer)
    config: Optional[Any] = None,             # FinetuneConfig, dict, or None (defaults)
    workspace: Optional[Any] = None,          # Workspace, root path, or None (resolve from env/cwd)
    logger: Optional[logging.Logger] = None   # Run logger; None = module logger
) -> Dict[str, Any]:  # The saved TrainingRunManifest as a dict
    """Run one segmentation finetune off a DatasetManifest.

    The recipe: warm-start the base model, attach a MultiLabelSegmentation
    task over the in-memory protocol (setup rebuilds the classifier head for
    the dataset's classes), fit, validate on the real-only holdout, and land
    weights + TrainingRunManifest under <workspace>/training-runs/<run_id>/."""
    log = logger or logging.getLogger(__name__)
    if not TRAIN_AVAILABLE:
        raise CapabilityFatalError("pyannote.audio training stack not installed.")
    cfg = config if isinstance(config, FinetuneConfig) else dict_to_config(FinetuneConfig, config or {})
    if not (0.0 < cfg.holdout_fraction < 0.5):
        raise CapabilityInputError(
            f"holdout_fraction {cfg.holdout_fraction} outside (0, 0.5) — the real-only "
            "holdout eval is part of the manifest contract (DEC 16159e09)",
            fields_invalid=["holdout_fraction"],
        )
    if isinstance(workspace, Workspace):
        ws = workspace
    elif workspace is not None:
        ws = resolve_workspace(explicit=Path(workspace))
    else:
        ws = resolve_workspace()
    if ws is None:
        raise CapabilityInputError(
            "No workspace resolved — training outputs land workspace-local (DEC e047beee); "
            "pass workspace= or set CJM_WORKSPACE",
            fields_invalid=["workspace"],
        )

    started = time.time()
    dataset, events, regions = load_dataset(dataset_manifest)
    classes = select_classes(dataset, cfg)
    spines = _eligible_spines(dataset)
    if not spines:
        raise CapabilityInputError(
            f"Dataset {dataset.get('dataset_id')} has no eligible spines to train on",
            fields_invalid=["dataset_manifest"],
        )

    run_id = new_training_run_id()
    run_dir = ws.root / TRAINING_RUNS_DIR / run_id
    run_dir.mkdir(parents=True)
    cache_dir = ws.root / TRAINING_RUNS_DIR / AUDIO_CACHE_DIR
    log.info(f"Training run {run_id}: dataset {dataset.get('dataset_id')}, "
             f"classes {classes}, {len(spines)} spine(s) -> {run_dir}")

    train_files: List[Dict[str, Any]] = []
    dev_files: List[Dict[str, Any]] = []
    spine_counts: List[Dict[str, Any]] = []
    for spine in spines:
        spine_key = (spine["source_id"], spine["skeleton_hash"])
        evs = [e for e in events if (e["source_id"], e["skeleton_hash"]) == spine_key]
        regs = [r for r in regions if (r["source_id"], r["skeleton_hash"]) == spine_key]
        if not evs:
            log.warning(f"Eligible spine {spine_key} contributed no events; skipping")
            continue
        source_path = evs[0]["source_path"]
        if cfg.prepare_wav:
            audio_path = _prepare_wav(source_path, evs[0].get("source_content_hash"),
                                      cache_dir, log)
        else:
            audio_path = Path(source_path)
        train_file, dev_file, counts = _build_spine_files(spine, evs, regs, classes, cfg, audio_path)
        train_files.append(train_file)
        dev_files.append(dev_file)
        spine_counts.append(counts)
    if not train_files:
        raise CapabilityInputError("No spine contributed training data", fields_invalid=["dataset_manifest"])

    # Re-order (and prune) the class list to upstream's encounter order — the
    # only order under which training targets land in the right columns
    ordered = encounter_class_order(train_files + dev_files)
    dropped = [c for c in classes if c not in ordered]
    if dropped:
        log.warning(f"Classes with no spans in any window dropped: {dropped}")
    classes = ordered
    for f in train_files + dev_files:
        f["classes"] = list(classes)

    protocol = _DatasetProtocol(
        train_files, dev_files,
        name=f"cjm-flywheel.Segmentation.{dataset.get('dataset_id', 'dataset')}",
    )
    seed_everything(cfg.seed, workers=True)
    try:
        model = Model.from_pretrained(cfg.base_model_id, token=_hf_token())
    except Exception as e:
        raise CapabilityFatalError(
            f"Failed to load {cfg.base_model_id!r}: {e} — gated model? Accept the model's "
            "user conditions on huggingface.co and provision a token in the worker env "
            "(hf auth login, or HF_TOKEN).",
        ) from e
    revision = _resolve_revision(cfg.base_model_id)

    task = MultiLabelSegmentation(
        protocol,
        classes=classes,
        duration=cfg.duration,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        cache=str(run_dir / "task_cache"),
    )
    model.task = task
    lr = cfg.learning_rate
    model.configure_optimizers = MethodType(
        lambda self: torch.optim.Adam(self.parameters(), lr=lr), model
    )

    accelerator, devices = _trainer_devices(cfg.device)
    trainer = Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=accelerator,
        devices=devices,
        default_root_dir=str(run_dir),
        logger=CSVLogger(str(run_dir), name="logs"),
        enable_checkpointing=False,  # the run's single artifact is the explicit final checkpoint
        enable_progress_bar=False,
        gradient_clip_val=0.5,
    )
    trainer.fit(model)
    val_metrics = trainer.validate(model)
    holdout = _holdout_eval(model, dev_files, classes, cfg)

    checkpoint = run_dir / "model.ckpt"
    trainer.save_checkpoint(str(checkpoint))
    content_hash = f"sha256:{_file_sha256(checkpoint)}"

    totals: Dict[str, Dict[str, int]] = {"train": {}, "dev": {}}
    for counts in spine_counts:
        for subset in ("train", "dev"):
            for label, n in counts[subset].items():
                totals[subset][label] = totals[subset].get(label, 0) + n

    manifest = TrainingRunManifest(
        run_id=run_id,
        created_at=started,
        config=config_to_dict(cfg),
        dataset_manifest=str(Path(dataset_manifest).resolve()),
        dataset_id=dataset.get("dataset_id", ""),
        base_model={"model_id": cfg.base_model_id, "revision": revision},
        environment=_environment_record(),
        classes=classes,
        files={"checkpoint": "model.ckpt", "logs": "logs"},
        artifact={
            "path": str(checkpoint),
            "content_hash": content_hash,
            "size_bytes": checkpoint.stat().st_size,
        },
        eval={
            "policy": {
                "holdout": "time-tail of each spine's annotated head",
                "fraction": cfg.holdout_fraction,
                "provenance_filter": "real",
                "reserved_tail": "above each spine's annotated_through — never read here; "
                                 "the live bench (DEC 8cf12c22)",
            },
            "metrics": {k: float(v) for k, v in (val_metrics[0] if val_metrics else {}).items()},
            "holdout": holdout,
        },
        counts={"spines": spine_counts, "totals": totals},
    )
    manifest.save(run_dir / "manifest.json", workspace=ws)
    log.info(f"Training run {run_id} complete: {checkpoint} ({content_hash[:19]}…), "
             f"eval {manifest.eval['metrics']}")
    return manifest.to_dict()


class PyannoteSegmentationFinetuneAdapter(ToolCapability):
    """Segmentation-finetune task adapter wrapping the pyannote training recipe.

    Capability-INTERNAL (DEC d02a38d4): this class IS the adapter contract's
    first trainable instance — `finetune(dataset_manifest, **kwargs)` keeps the
    surface-match PREFIX RULE shape so the generic interface lib (minted at
    n=2 trainable capabilities) binds without a signature change. Long runs
    ride the same worker regime as long inference runs."""

    config_class = FinetuneConfig

    def __init__(self):
        """Initialize the segmentation-finetune adapter."""
        self.logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self.config: Optional[FinetuneConfig] = None

    @property
    def name(self) -> str:  # Capability name identifier
        """Capability identity, derived from the installed distribution (PILLAR 1c)."""
        from importlib.metadata import metadata, packages_distributions
        dist = (packages_distributions().get(__package__) or [__package__.replace("_", "-")])[0]
        return metadata(dist)["Name"]

    @property
    def version(self) -> str:  # Capability version string
        """Get the capability version string."""
        from cjm_capability_pyannote import __version__
        return __version__

    def get_current_config(self) -> Dict[str, Any]:  # Current configuration as dictionary
        """Return current configuration state."""
        return config_to_dict(self.config) if self.config else {}

    def get_config_schema(self) -> Dict[str, Any]:  # JSON Schema for configuration
        """Return JSON Schema for UI generation."""
        return dataclass_to_jsonschema(FinetuneConfig)

    def _apply_config(
        self,
        config: Optional[Any] = None  # Configuration dataclass, dict, or None
    ) -> None:
        """Apply config values (no heavy-resource work — the adapter holds no
        loaded state between runs; every finetune loads its own base model)."""
        self.config = dict_to_config(FinetuneConfig, config or {})

    def initialize(
        self,
        config: Optional[Any] = None  # Configuration dataclass, dict, or None
    ) -> None:
        """First-time setup."""
        self._apply_config(config)
        self.logger.info(f"Initialized segmentation-finetune adapter "
                         f"(base_model={self.config.base_model_id}, device={self.config.device})")

    def finetune(
        self,
        dataset_manifest: str,  # Path to the consumed DatasetManifest json
        **kwargs                # workspace override + provenance pass-through
    ) -> Dict[str, Any]:  # The saved TrainingRunManifest as a dict
        """Finetune the configured base model on a dataset (manifest pointer in,
        manifest dict out)."""
        if not isinstance(dataset_manifest, str) or not dataset_manifest:
            raise CapabilityInputError(
                f"Unsupported dataset_manifest input: {dataset_manifest!r}; expected a "
                "non-empty path string",
                fields_invalid=["dataset_manifest"],
            )
        if not os.path.exists(dataset_manifest):
            raise CapabilityInputError(
                f"Dataset manifest does not exist: {dataset_manifest}",
                fields_invalid=["dataset_manifest"],
            )
        if self.config is None:
            self._apply_config(None)
        return run_finetune(dataset_manifest, config=self.config,
                            workspace=kwargs.get("workspace"), logger=self.logger)

    def is_available(self) -> bool:  # True if the training stack is importable
        """Check if the pyannote training stack is available."""
        return TRAIN_AVAILABLE

    def cleanup(self) -> None:
        """Release resources on unload (no persistent state between runs)."""


def main(argv: Optional[List[str]] = None) -> int:
    """Dev runner: `python -m cjm_capability_pyannote.finetune <manifest> [...]`.

    The worker regime carries production training runs (DEC d02a38d4); this
    entry exists so a run is launchable straight off a dataset directory."""
    parser = argparse.ArgumentParser(
        prog="cjm_capability_pyannote.finetune",
        description="Finetune pyannote segmentation on a correction-core DatasetManifest.",
    )
    parser.add_argument("dataset_manifest", help="Path to the DatasetManifest json")
    parser.add_argument("--workspace", help="Workspace root (default: CJM_WORKSPACE / upward walk)")
    parser.add_argument("--base-model-id", dest="base_model_id")
    parser.add_argument("--device")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    parser.add_argument("--max-epochs", dest="max_epochs", type=int)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float)
    parser.add_argument("--num-workers", dest="num_workers", type=int)
    parser.add_argument("--holdout-fraction", dest="holdout_fraction", type=float)
    parser.add_argument("--min-class-count", dest="min_class_count", type=int)
    parser.add_argument("--exclude-labels", dest="exclude_labels", nargs="+",
                        help="Labels never trained as classes (hard negatives); "
                             "surfaced for the f2d15413 promotion decision — labels "
                             "past --min-class-count auto-train unless listed here")
    parser.add_argument("--no-speech-class", dest="include_speech_class", action="store_false", default=None)
    parser.add_argument("--no-prepare-wav", dest="prepare_wav", action="store_false", default=None)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    overrides = {k: v for k, v in vars(args).items()
                 if k not in ("dataset_manifest", "workspace") and v is not None}
    result = run_finetune(args.dataset_manifest,
                          config=dict_to_config(FinetuneConfig, overrides),
                          workspace=args.workspace)
    print(json.dumps({"run_id": result["run_id"],
                      "artifact": result["artifact"],
                      "classes": result["classes"],
                      "eval": result["eval"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
