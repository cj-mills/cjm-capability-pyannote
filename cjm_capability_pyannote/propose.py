"""Event-proposal generation (flywheel leg 4): a TrainingRunManifest in, a
workspace-local PROPOSAL SET out — durable inference-run output per DEC
8e05b87b (verdicts are NEVER stored: they DERIVE later from joining this set
against final spine state; the accept gesture IS the insert op). Capability-
INTERNAL, CLI-first per DEC a5aa43b9; the manifest chain extends per DEC
16159e09: dataset manifest -> training-run manifest -> THIS -> spine-state
join -> bench verdicts."""

import argparse
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from cjm_capability_pyannote.finetune import (_file_sha256, _prepare_wav, AUDIO_CACHE_DIR,
                                              TRAINING_RUNS_DIR)
from cjm_substrate.core.errors import CapabilityFatalError, CapabilityInputError
from cjm_substrate.core.workspace import (relativize_recorded, resolve_recorded_tree,
                                          resolve_workspace, Workspace)
from cjm_substrate.utils.validation import config_to_dict, dict_to_config, SCHEMA_DESC, SCHEMA_TITLE

# Inference Imports (heavy; shares the finetune module's availability contract)
try:
    import torch
    from pyannote.audio import Inference, Model
    from pyannote.core import Segment
    from pyannote.audio.core.io import Audio
    PROPOSE_AVAILABLE = True
except ImportError:
    PROPOSE_AVAILABLE = False

# Format tag of the consumed training-run manifest (the model pointer)
TRAINING_RUN_MANIFEST_FORMAT = "cjm-capability-pyannote/training-run-manifest"

# Workspace-local landing zone for proposal sets (an inference run is a run —
# the datasets/ + training-runs/ siblings' pattern, DEC a5883992)
PROPOSALS_DIR = "proposals"


def new_proposal_set_id() -> str:  # e.g. "propset_20260729_180000_1a2b3c4d"
    """Generate a unique, sortable proposal-set id (the new_run_id pattern, propset kind)."""
    return f"propset_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


@dataclass
class ProposalSetManifest:
    """Durable record of one proposal-generation run (chainable; DEC 16159e09).

    Format `cjm-capability-pyannote/proposal-set-manifest`. The proposal set is
    the durable half of the DERIVED-VERDICTS contract (DEC 8e05b87b): accepts
    materialize as insert ops on the spine, edits as nudges, rejects stay
    unmarked — the bench joins THIS set against final spine state, so the set
    itself must record exactly what was proposed, from which model, over which
    window, at which thresholds."""
    proposal_set_id: str     # Unique proposal-set identifier (sortable, run-id pattern)
    created_at: float        # Unix timestamp at generation start
    config: Dict[str, Any]   # Threshold/window snapshot (ProposeConfig as dict)
    training_run_manifest: str  # Recorded path of the consumed TrainingRunManifest (the pointer)
    training_run_id: str     # Consumed training-run id (join key for manifest-chain queries)
    model: Dict[str, Any] = field(default_factory=dict)   # Model identity: artifact content hash + trained classes
    source: Dict[str, Any] = field(default_factory=dict)  # Audio identity: path + content hash + optional spine join keys
    window: Dict[str, float] = field(default_factory=dict)  # Proposed-over window [start, end] in source seconds
    classes: List[str] = field(default_factory=list)      # Classes proposals were generated for
    files: Dict[str, str] = field(default_factory=dict)   # Set-relative data files (proposals)
    counts: Dict[str, int] = field(default_factory=dict)  # Proposals per class

    FORMAT: str = field(default="cjm-capability-pyannote/proposal-set-manifest", repr=False)  # Format tag
    VERSION: str = field(default="0.1.0", repr=False)                                          # Schema version

    def to_dict(self) -> Dict[str, Any]:  # Plain-dict form for JSON serialization
        """Serialize to a plain dict."""
        return {
            "format": self.FORMAT,
            "version": self.VERSION,
            "proposal_set_id": self.proposal_set_id,
            "created_at": self.created_at,
            "config": self.config,
            "training_run_manifest": self.training_run_manifest,
            "training_run_id": self.training_run_id,
            "model": self.model,
            "source": self.source,
            "window": dict(self.window),
            "classes": list(self.classes),
            "files": dict(self.files),
            "counts": dict(self.counts),
        }

    def save(
        self,
        path: Union[str, Path],  # Destination JSON file (parent dirs created)
        workspace=None,  # Active Workspace; owned paths record as ${WS}/<rel> (5daadfc4 rung f)
    ) -> Path:  # The written path
        """Write the manifest as pretty-printed JSON (WS-token recorded paths)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(relativize_recorded(self.to_dict(), workspace), indent=2))
        return out


@dataclass
class ProposeConfig:
    """Configuration for one proposal-generation run.

    Thresholds are the OPERATING POINT: they land verbatim in the manifest so
    the bench can attribute accept/reject rates to the exact point chosen."""

    classes: List[str] = field(
        default_factory=lambda: ["inhale"],
        metadata={
            SCHEMA_TITLE: "Classes",
            SCHEMA_DESC: "Event classes to propose (must be classes the model was trained on).",
        }
    )
    onset: float = field(
        default=0.5,
        metadata={
            SCHEMA_TITLE: "Onset Threshold",
            SCHEMA_DESC: "Frame probability at which a span opens (hysteresis high-water mark).",
        }
    )
    offset: float = field(
        default=0.5,
        metadata={
            SCHEMA_TITLE: "Offset Threshold",
            SCHEMA_DESC: "Frame probability below which an open span closes (hysteresis low-water mark).",
        }
    )
    min_duration_on: float = field(
        default=0.05,
        metadata={
            SCHEMA_TITLE: "Min Span Duration",
            SCHEMA_DESC: "Spans shorter than this many seconds are dropped.",
        }
    )
    min_duration_off: float = field(
        default=0.05,
        metadata={
            SCHEMA_TITLE: "Min Gap Duration",
            SCHEMA_DESC: "Adjacent spans separated by less than this many seconds merge.",
        }
    )
    duration: float = field(
        default=10.0,
        metadata={
            SCHEMA_TITLE: "Inference Chunk Duration",
            SCHEMA_DESC: "Sliding-window chunk length in seconds (match the training chunk).",
        }
    )
    batch_size: int = field(
        default=32,
        metadata={SCHEMA_TITLE: "Batch Size", SCHEMA_DESC: "Inference chunks per batch."}
    )
    device: str = field(
        default="auto",
        metadata={
            SCHEMA_TITLE: "Device",
            SCHEMA_DESC: "Inference device: 'auto' (cuda when available), 'cuda', 'cuda:N', or 'cpu'.",
        }
    )
    prepare_wav: bool = field(
        default=True,
        metadata={
            SCHEMA_TITLE: "Prepare WAV",
            SCHEMA_DESC: "Transcode the source once into the shared 16 kHz mono wav cache "
                         "(same cache the finetune runs use).",
        }
    )


def load_training_run(
    manifest_path: Union[str, Path]  # TrainingRunManifest json (or its run directory)
) -> Tuple[Dict[str, Any], Path]:  # (resolved manifest, checkpoint path)
    """Load a TrainingRunManifest and resolve its checkpoint artifact."""
    p = Path(manifest_path)
    if p.is_dir():
        p = p / "manifest.json"
    p = p.resolve()
    if not p.is_file():
        raise CapabilityInputError(
            f"Training-run manifest not found: {p}",
            fields_invalid=["training_run"],
        )
    data = json.loads(p.read_text())
    fmt = data.get("format")
    if fmt != TRAINING_RUN_MANIFEST_FORMAT:
        raise CapabilityInputError(
            f"{p} is not a training-run manifest (format {fmt!r}; expected "
            f"{TRAINING_RUN_MANIFEST_FORMAT!r})",
            fields_invalid=["training_run"],
        )
    resolved = resolve_recorded_tree(data, p)
    checkpoint = Path(resolved.get("artifact", {}).get("path", ""))
    if not checkpoint.is_file():
        raise CapabilityInputError(
            f"Training-run checkpoint not found: {checkpoint} (recorded in {p})",
            fields_invalid=["training_run"],
        )
    return resolved, checkpoint


def spans_from_scores(
    times: List[float],     # Frame-center times (seconds, monotonically increasing)
    step: float,            # Frame step in seconds (span edges extend half a step past centers)
    probs: List[float],     # Per-frame probabilities for ONE class
    onset: float,           # Open threshold (hysteresis high)
    offset: float,          # Close threshold (hysteresis low)
    min_duration_on: float,   # Drop spans shorter than this
    min_duration_off: float,  # Merge spans separated by less than this
) -> List[Tuple[float, float, float]]:  # (start, end, max-prob score) per span
    """Extract event spans from frame scores by onset/offset hysteresis.

    Transparent pure fold (no pyannote Binarize dependency): a span opens when
    probability reaches `onset`, closes when it drops below `offset`; close
    gaps merge (max score carries), short spans drop. Span edges extend half a
    frame step past the first/last active frame centers."""
    half = step / 2.0
    raw: List[Tuple[float, float, float]] = []
    open_start: Optional[float] = None
    peak = 0.0
    for t, p in zip(times, probs):
        if open_start is None:
            if p >= onset:
                open_start, peak = t - half, p
        else:
            if p < offset:
                raw.append((open_start, t - half, peak))
                open_start = None
            else:
                peak = max(peak, p)
    if open_start is not None and times:
        raw.append((open_start, times[-1] + half, peak))

    merged: List[Tuple[float, float, float]] = []
    for start, end, score in raw:
        if merged and start - merged[-1][1] < min_duration_off:
            prev_start, _prev_end, prev_score = merged[-1]
            merged[-1] = (prev_start, end, max(prev_score, score))
        else:
            merged.append((start, end, score))
    return [(s, e, score) for s, e, score in merged if e - s >= min_duration_on]


def run_propose(
    training_run: Union[str, Path],           # TrainingRunManifest json or run directory (the model pointer)
    source: Union[str, Path],                 # Source audio to propose over (spine coordinates)
    config: Optional[Any] = None,             # ProposeConfig, dict, or None (defaults)
    start: float = 0.0,                       # Window start in source seconds (the watermark for a reserved-tail run)
    end: Optional[float] = None,              # Window end in source seconds; None = end of source
    source_id: Optional[str] = None,          # Optional spine join key, recorded verbatim
    skeleton_hash: Optional[str] = None,      # Optional spine join key, recorded verbatim
    workspace: Optional[Any] = None,          # Workspace, root path, or None (resolve from env/cwd)
    logger: Optional[logging.Logger] = None   # Run logger; None = module logger
) -> Dict[str, Any]:  # The saved ProposalSetManifest as a dict
    """Generate event-span proposals over a source window.

    The window is recorded, never inferred: for the reserved-tail bench the
    caller passes the spine's annotated_through watermark as `start` — the
    generator itself stays a generic detection tool (gate semantics live with
    the workflow, DEC 8e05b87b)."""
    log = logger or logging.getLogger(__name__)
    if not PROPOSE_AVAILABLE:
        raise CapabilityFatalError("pyannote.audio inference stack not installed.")
    cfg = config if isinstance(config, ProposeConfig) else dict_to_config(ProposeConfig, config or {})
    source_path = Path(source)
    if not source_path.is_file():
        raise CapabilityInputError(
            f"Source audio does not exist: {source_path}",
            fields_invalid=["source"],
        )
    if isinstance(workspace, Workspace):
        ws = workspace
    elif workspace is not None:
        ws = resolve_workspace(explicit=Path(workspace))
    else:
        ws = resolve_workspace()
    if ws is None:
        raise CapabilityInputError(
            "No workspace resolved — proposal sets land workspace-local; "
            "pass workspace= or set CJM_WORKSPACE",
            fields_invalid=["workspace"],
        )

    started = time.time()
    run_manifest, checkpoint = load_training_run(training_run)
    trained_classes = list(run_manifest.get("classes", []))
    unknown = [c for c in cfg.classes if c not in trained_classes]
    if unknown:
        raise CapabilityInputError(
            f"Model {run_manifest.get('run_id')} was not trained on {unknown} "
            f"(trained classes: {trained_classes})",
            fields_invalid=["classes"],
        )

    content_hash = f"sha256:{_file_sha256(source_path)}"
    cache_dir = ws.root / TRAINING_RUNS_DIR / AUDIO_CACHE_DIR
    if cfg.prepare_wav:
        audio_path = _prepare_wav(str(source_path), content_hash, cache_dir, log)
    else:
        audio_path = source_path
    if end is None:
        end = float(Audio().get_duration({"audio": str(audio_path)}))
    if not (0.0 <= start < end):
        raise CapabilityInputError(
            f"Window [{start}, {end}] is empty or inverted",
            fields_invalid=["start", "end"],
        )

    model = Model.from_pretrained(str(checkpoint))
    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    inference = Inference(model, duration=cfg.duration, step=cfg.duration / 2.0,
                          batch_size=cfg.batch_size, device=torch.device(device))
    log.info(f"Proposing {cfg.classes} over [{start:.1f}, {end:.1f}]s of {source_path.name} "
             f"with {run_manifest.get('run_id')} on {device}")
    scores = inference.crop({"audio": str(audio_path)}, Segment(start, end))
    frames = scores.sliding_window
    times = [frames[i].middle for i in range(scores.data.shape[0])]

    set_id = new_proposal_set_id()
    set_dir = ws.root / PROPOSALS_DIR / set_id
    set_dir.mkdir(parents=True)
    counts: Dict[str, int] = {}
    with open(set_dir / "proposals.jsonl", "w") as f:
        for label in cfg.classes:
            column = trained_classes.index(label)
            spans = spans_from_scores(
                times, frames.step, [float(v) for v in scores.data[:, column]],
                onset=cfg.onset, offset=cfg.offset,
                min_duration_on=cfg.min_duration_on, min_duration_off=cfg.min_duration_off,
            )
            counts[label] = len(spans)
            for span_start, span_end, score in spans:
                f.write(json.dumps({
                    "proposal_id": str(uuid.uuid4()),
                    "label": label,
                    "start_time": round(max(start, span_start), 4),
                    "end_time": round(min(end, span_end), 4),
                    "score": round(score, 4),
                }) + "\n")

    manifest = ProposalSetManifest(
        proposal_set_id=set_id,
        created_at=started,
        config=config_to_dict(cfg),
        training_run_manifest=str((Path(training_run) / "manifest.json").resolve()
                                  if Path(training_run).is_dir() else Path(training_run).resolve()),
        training_run_id=run_manifest.get("run_id", ""),
        model={"content_hash": run_manifest.get("artifact", {}).get("content_hash"),
               "classes": trained_classes},
        source={"path": str(source_path.resolve()), "content_hash": content_hash,
                **({"source_id": source_id} if source_id else {}),
                **({"skeleton_hash": skeleton_hash} if skeleton_hash else {})},
        window={"start": float(start), "end": float(end)},
        classes=list(cfg.classes),
        files={"proposals": "proposals.jsonl"},
        counts=counts,
    )
    manifest.save(set_dir / "manifest.json", workspace=ws)
    log.info(f"Proposal set {set_id} complete: {counts} -> {set_dir}")
    return manifest.to_dict()


def main(argv: Optional[List[str]] = None) -> int:
    """Dev runner: `python -m cjm_capability_pyannote.propose <training-run> <audio> [...]`.

    CLI-first per DEC a5aa43b9; the reserved-tail bench run passes the spine's
    watermark as --start and the spine join keys for the later verdict join."""
    parser = argparse.ArgumentParser(
        prog="cjm_capability_pyannote.propose",
        description="Generate event-span proposals from a finetuned segmentation model.",
    )
    parser.add_argument("training_run", help="TrainingRunManifest json (or its run directory)")
    parser.add_argument("source", help="Source audio to propose over")
    parser.add_argument("--workspace", help="Workspace root (default: CJM_WORKSPACE / upward walk)")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Window start in seconds (the watermark for a reserved-tail run)")
    parser.add_argument("--end", type=float, help="Window end in seconds (default: end of source)")
    parser.add_argument("--source-id", dest="source_id", help="Spine join key (recorded verbatim)")
    parser.add_argument("--skeleton-hash", dest="skeleton_hash", help="Spine join key (recorded verbatim)")
    parser.add_argument("--classes", nargs="+")
    parser.add_argument("--onset", type=float)
    parser.add_argument("--offset", type=float)
    parser.add_argument("--min-duration-on", dest="min_duration_on", type=float)
    parser.add_argument("--min-duration-off", dest="min_duration_off", type=float)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--no-prepare-wav", dest="prepare_wav", action="store_false", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config_keys = {f.name for f in ProposeConfig.__dataclass_fields__.values()}
    overrides = {k: v for k, v in vars(args).items() if k in config_keys and v is not None}
    result = run_propose(
        args.training_run, args.source,
        config=dict_to_config(ProposeConfig, overrides),
        start=args.start, end=args.end,
        source_id=args.source_id, skeleton_hash=args.skeleton_hash,
        workspace=args.workspace,
    )
    print(json.dumps({"proposal_set_id": result["proposal_set_id"],
                      "counts": result["counts"],
                      "window": result["window"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
