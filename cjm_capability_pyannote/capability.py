"""Speaker-diarization tool capability using pyannote.audio (session-D DECs 18d7de80 + d6df3a8e): source audio in, ANONYMOUS time-ranged speaker turns out — the machine half of speaker assignment; identity stays the correction TUI's human lane (DEC 44afb2df)."""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from cjm_capability_primitives.speaker_diarization import SpeakerDiarizationResult, SpeakerTurn
from cjm_substrate.core.capability import RELOAD_TRIGGER, ToolCapability
from cjm_substrate.core.errors import CapabilityFatalError, CapabilityInputError
from cjm_substrate.utils.validation import (config_to_dict, dataclass_to_jsonschema, dict_to_config,
                                            SCHEMA_DESC, SCHEMA_TITLE)
from cjm_substrate_torch_utils.memory import release_model

# The tool wraps a pyannote.audio Pipeline (community-1 by default). Models are
# HF-GATED: accepting the model's user conditions + a HuggingFace token are ENV
# PROVISIONING facts (hf auth login, or HF_TOKEN/HUGGING_FACE_HUB_TOKEN in the
# worker env) — never config knobs. First diarize() downloads the pipeline into
# the standard HF cache.

# pyannote Imports
try:
    import torch
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False


@dataclass
class PyannoteConfig:
    """Configuration for the pyannote diarization pipeline.

    Speaker-count hints are deliberately NOT config fields: they are
    per-CALL knowledge (`diarize(..., num_speakers=2)`) — how many people sat
    in one recording is a property of the source, not of the tool."""

    model_id: str = field(
        default="pyannote/speaker-diarization-community-1",
        metadata={
            SCHEMA_TITLE: "Model ID",
            SCHEMA_DESC: "HuggingFace pipeline id (gated: accept the model's user "
                         "conditions and provision a token in the worker env).",
            RELOAD_TRIGGER: "pipeline",  # a model change invalidates the loaded pipeline
        }
    )
    device: str = field(
        default="auto",
        metadata={
            SCHEMA_TITLE: "Device",
            SCHEMA_DESC: "Torch device for the pipeline: 'auto' (cuda when available), "
                         "'cuda', 'cuda:N', or 'cpu'.",
            RELOAD_TRIGGER: "pipeline",  # a device change invalidates the loaded pipeline
        }
    )


class PyannoteDiarizationCapability(ToolCapability):
    """Speaker-diarization tool capability wrapping a pyannote.audio Pipeline.

    Native-surface model (PILLAR 1c): `diarize` runs the pretrained pipeline
    over the caller's ORIGINAL source audio and returns typed `SpeakerTurn`s
    in that audio's coordinates, time-ordered; overlapping turns are signal
    (overlapped speech), not an error. Cluster labels (SPEAKER_00, ...) are
    result-scoped — the assignment lane's cluster-name-once verdict binds
    them to identities (DEC 8a4df244). The generic adapter
    (cjm-speaker-diarization-adapter-interface) is a thin pass-through."""

    config_class = PyannoteConfig

    def __init__(self):
        """Initialize the pyannote diarization capability."""
        self.logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self.config: Optional[PyannoteConfig] = None
        self._pipeline = None
        self._loaded_device: Optional[str] = None

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
        return dataclass_to_jsonschema(PyannoteConfig)

    def _apply_config(
        self,
        config: Optional[Any] = None  # Configuration dataclass, dict, or None
    ) -> None:
        """Apply config values only (no heavy-resource work). Called by
        initialize (first-time) and by the substrate's reconfigure delta path."""
        self.config = dict_to_config(PyannoteConfig, config or {})

    def initialize(
        self,
        config: Optional[Any] = None  # Configuration dataclass, dict, or None
    ) -> None:
        """First-time setup. Config application is factored into _apply_config;
        the substrate's reconfigure(old, new) fires _release_pipeline on a
        model_id/device change (RELOAD_TRIGGER) then re-applies config."""
        self._apply_config(config)
        self.logger.info(f"Initialized pyannote diarization capability "
                         f"(model={self.config.model_id}, device={self.config.device})")

    @staticmethod
    def _hf_token() -> Optional[str]:
        """The HF token from the worker env (gated-model pull); None falls back
        to the hub's cached login (hf auth login) — provisioning, not config."""
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    def _load_pipeline(self) -> None:
        """Lazily load the pretrained pipeline and move it to the resolved device.

        Gated-model failures (no token / conditions not accepted) surface as
        CapabilityFatalError pointing at the provisioning fix — the operator
        acts in the ENV, not in config."""
        if self._pipeline is not None:
            return
        if not PYANNOTE_AVAILABLE:
            raise CapabilityFatalError(  # load-time dependency missing — fatal until operator installs pyannote.audio
                "pyannote.audio not installed.",
            )
        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Loading pipeline {self.config.model_id} on {device}")
        try:
            pipeline = Pipeline.from_pretrained(self.config.model_id,
                                                token=self._hf_token())
        except Exception as e:
            raise CapabilityFatalError(
                f"Failed to load {self.config.model_id!r}: {e} — gated model? "
                "Accept the model's user conditions on huggingface.co and provision "
                "a token in the worker env (hf auth login, or HF_TOKEN).",
            ) from e
        if pipeline is None:
            raise CapabilityFatalError(
                f"Pipeline.from_pretrained returned None for {self.config.model_id!r} "
                "— the model is gated: accept its user conditions and provision a token.",
            )
        self._pipeline = pipeline.to(torch.device(device))
        self._loaded_device = device

    def _release_pipeline(self) -> None:
        """Release the loaded pipeline (GPU memory back). RELOAD_TRIGGER target
        for model_id/device; on_disable / cleanup delegate here."""
        release_model(self, ["_pipeline"], device="cuda", logger=self.logger)
        self._pipeline = None
        self._loaded_device = None

    def diarize(
        self,
        audio: str,                          # Path to the source audio to diarize (turn times index into THIS audio)
        num_speakers: Optional[int] = None,  # Exact speaker count, when the source is known (per-call knowledge)
        min_speakers: Optional[int] = None,  # Lower bound hint
        max_speakers: Optional[int] = None,  # Upper bound hint
        **kwargs                             # Provenance pass-through (unused by diarization compute)
    ) -> SpeakerDiarizationResult:  # Time-ordered anonymous speaker turns
        """Diarize source audio into time-ordered anonymous speaker turns.

        Iterates the pipeline output via `itertracks(yield_label=True)` — the
        stable surface across pyannote.audio 3.x (bare Annotation) and 4.x
        (DiarizeOutput.speaker_diarization). Turns keep source coordinates;
        overlapping turns pass through untouched (overlapped speech is
        signal the downstream join wants)."""
        if not isinstance(audio, str) or not audio:
            raise CapabilityInputError(  # typed input-validation
                f"Unsupported audio input: {audio!r}; expected a non-empty path string",
                fields_invalid=["audio"],
            )
        if not os.path.exists(audio):
            raise CapabilityInputError(
                f"Audio path does not exist: {audio}",
                fields_invalid=["audio"],
            )
        self._load_pipeline()

        hints = {k: v for k, v in (("num_speakers", num_speakers),
                                   ("min_speakers", min_speakers),
                                   ("max_speakers", max_speakers)) if v is not None}
        output = self._pipeline(audio, **hints)
        annotation = getattr(output, "speaker_diarization", output)
        turns: List[SpeakerTurn] = [
            SpeakerTurn(start=float(segment.start), end=float(segment.end),
                        speaker=str(label))
            for segment, _track, label in annotation.itertracks(yield_label=True)
        ]
        turns.sort(key=lambda t: (t.start, t.end))
        return SpeakerDiarizationResult(
            turns=turns,
            metadata={
                "model_id": self.config.model_id,
                "device": self._loaded_device,
                "speaker_count": len({t.speaker for t in turns}),
                "turn_count": len(turns),
                **({"hints": hints} if hints else {}),
            },
        )

    def is_available(self) -> bool:  # True if pyannote.audio is importable
        """Check if pyannote.audio is available."""
        return PYANNOTE_AVAILABLE

    def prefetch(self) -> None:
        """Eagerly load the pipeline (model download + device move) so the
        first diarize() doesn't pay the cold-load cost."""
        self._load_pipeline()

    def on_disable(self) -> None:
        """Release the pipeline when the operator disables the capability (worker stays alive)."""
        self._release_pipeline()

    def cleanup(self) -> None:
        """Release resources on unload."""
        self._release_pipeline()
