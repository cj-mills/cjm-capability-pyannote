"""Tests for the pyannote diarization capability — config plumbing, input
validation, protocol conformance, and the turn-mapping contract (model-free:
the pipeline is faked; the real gated model exercises via tests_manual/)."""
import pytest

from cjm_capability_primitives.speaker_diarization import SpeakerDiarizationResult
from cjm_capability_pyannote.capability import PyannoteConfig, PyannoteDiarizationCapability
from cjm_speaker_diarization_adapter_interface.adapter import SpeakerDiarizationToolProtocol
from cjm_substrate.core.errors import CapabilityInputError


class _FakeSegment:
    def __init__(self, start, end):
        self.start, self.end = start, end


class _FakeAnnotation:
    """Stands in for pyannote.core.Annotation: itertracks(yield_label=True)."""
    def __init__(self, tracks):
        self._tracks = tracks

    def itertracks(self, yield_label=False):
        assert yield_label
        return iter(self._tracks)


class _FakeOutput:
    """Stands in for the 4.x DiarizeOutput (bare Annotation = the 3.x shape)."""
    def __init__(self, annotation):
        self.speaker_diarization = annotation


@pytest.fixture
def cap():
    c = PyannoteDiarizationCapability()
    c.initialize()
    return c


def test_satisfies_tool_protocol(cap):
    assert isinstance(cap, SpeakerDiarizationToolProtocol)


def test_config_defaults_and_schema(cap):
    cfg = cap.get_current_config()
    assert cfg["model_id"] == "pyannote/speaker-diarization-community-1"
    assert cfg["device"] == "auto"
    schema = cap.get_config_schema()
    assert "model_id" in schema["properties"] and "device" in schema["properties"]


def test_bad_audio_input_is_typed_error(cap):
    with pytest.raises(CapabilityInputError):
        cap.diarize("")
    with pytest.raises(CapabilityInputError):
        cap.diarize(123)
    with pytest.raises(CapabilityInputError):
        cap.diarize("/nonexistent/audio.wav")


def test_turn_mapping_orders_and_counts(cap, tmp_path):
    """The mapping contract: itertracks -> typed turns, time-ordered, labels
    stringified, speaker/turn counts + hints in metadata; works on BOTH the
    4.x DiarizeOutput shape and the 3.x bare-Annotation shape."""
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"\x00")
    tracks = [(_FakeSegment(1.8, 3.9), "A", "SPEAKER_01"),
              (_FakeSegment(0.2, 1.5), "B", "SPEAKER_00"),
              (_FakeSegment(4.2, 5.7), "C", "SPEAKER_00")]

    class _FakePipeline:
        def __init__(self, wrap):
            self.wrap = wrap
            self.calls = []

        def __call__(self, audio, **hints):
            self.calls.append(hints)
            ann = _FakeAnnotation(tracks)
            return _FakeOutput(ann) if self.wrap else ann

    for wrap in (True, False):
        cap._pipeline = _FakePipeline(wrap)
        cap._loaded_device = "cpu"
        result = cap.diarize(str(wav), num_speakers=2)
        assert isinstance(result, SpeakerDiarizationResult)
        assert [(t.start, t.speaker) for t in result.turns] == [
            (0.2, "SPEAKER_00"), (1.8, "SPEAKER_01"), (4.2, "SPEAKER_00")]
        assert result.metadata["speaker_count"] == 2
        assert result.metadata["turn_count"] == 3
        assert result.metadata["hints"] == {"num_speakers": 2}
        assert cap._pipeline.calls == [{"num_speakers": 2}]


def test_release_is_idempotent(cap):
    cap._release_pipeline()
    cap._release_pipeline()
    assert cap._pipeline is None
