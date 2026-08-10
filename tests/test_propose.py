"""Tests for proposal generation — the hysteresis span fold, manifest shape,
and typed input errors (model-free: no inference run; the real recipe
exercises via the dev runner off a real training run)."""
import json

import pytest

from cjm_capability_pyannote.propose import (load_training_run, new_proposal_set_id,
                                             ProposalSetManifest, ProposeConfig, run_propose,
                                             spans_from_scores, tier2_extras)
from cjm_substrate.core.errors import CapabilityInputError
from cjm_substrate.core.workspace import Workspace


def _spans(probs, **kw):
    step = 0.1
    times = [i * step + step / 2 for i in range(len(probs))]
    defaults = dict(onset=0.5, offset=0.5, min_duration_on=0.0, min_duration_off=0.0)
    defaults.update(kw)
    return spans_from_scores(times, step, probs, **defaults)


def test_spans_basic_open_close():
    spans = _spans([0.1, 0.9, 0.9, 0.1, 0.1])
    assert len(spans) == 1
    start, end, score = spans[0]
    assert start == pytest.approx(0.1) and end == pytest.approx(0.3)
    assert score == pytest.approx(0.9)


def test_spans_hysteresis_offset_below_onset():
    # opens at 0.8 (>= onset 0.6), stays open through 0.5 (>= offset 0.4)
    spans = _spans([0.8, 0.5, 0.5, 0.3], onset=0.6, offset=0.4)
    assert len(spans) == 1
    assert spans[0][1] == pytest.approx(0.3 + 0.05 - 0.05)  # closes entering the 0.3 frame


def test_spans_open_at_end_closes_at_last_frame():
    spans = _spans([0.1, 0.9, 0.9])
    assert len(spans) == 1
    assert spans[0][1] == pytest.approx(0.3)  # last frame center + half step


def test_spans_merge_and_min_duration():
    probs = [0.9, 0.1, 0.9, 0.1]
    # gap of one frame (0.1s) merges when min_duration_off > 0.1
    merged = _spans(probs, min_duration_off=0.15)
    assert len(merged) == 1 and merged[0][2] == pytest.approx(0.9)
    apart = _spans(probs, min_duration_off=0.05)
    assert len(apart) == 2
    # both spans are 0.1s: min_duration_on above that drops them
    assert _spans(probs, min_duration_on=0.2) == []


def test_spans_empty_and_all_low():
    assert _spans([]) == []
    assert _spans([0.1, 0.2, 0.3]) == []


def test_proposal_manifest_shape_and_ws_tokens(tmp_path):
    ws = Workspace(root=tmp_path)
    m = ProposalSetManifest(
        proposal_set_id="propset_x", created_at=1.0, config={"onset": 0.5},
        training_run_manifest=str(tmp_path / "training-runs" / "t" / "manifest.json"),
        training_run_id="trainrun_t",
    )
    out = m.save(tmp_path / "proposals" / "propset_x" / "manifest.json", workspace=ws)
    data = json.loads(out.read_text())
    assert data["format"] == "cjm-capability-pyannote/proposal-set-manifest"
    assert data["version"] == "0.2.0"
    assert data["training_run_manifest"] == "${WS}/training-runs/t/manifest.json"
    # Single-tier set: the tier2_counts key stays OUT of the serialized shape.
    assert "tier2_counts" not in data
    dual = ProposalSetManifest(
        proposal_set_id="propset_y", created_at=1.0,
        config={"onset": 0.5, "tier2_onset": 0.35},
        training_run_manifest=str(tmp_path / "training-runs" / "t" / "manifest.json"),
        training_run_id="trainrun_t", tier2_counts={"inhale": 3},
    )
    assert dual.to_dict()["tier2_counts"] == {"inhale": 3}


def test_load_training_run_rejects_wrong_format(tmp_path):
    bad = tmp_path / "manifest.json"
    bad.write_text(json.dumps({"format": "something-else"}))
    with pytest.raises(CapabilityInputError):
        load_training_run(bad)
    with pytest.raises(CapabilityInputError):
        load_training_run(tmp_path / "missing" / "manifest.json")


def test_run_propose_input_errors(tmp_path, monkeypatch):
    monkeypatch.delenv("CJM_WORKSPACE", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(CapabilityInputError):
        run_propose(tmp_path / "t", tmp_path / "missing.wav")  # source missing
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"\x00")
    with pytest.raises(CapabilityInputError):
        run_propose(tmp_path / "t", audio)  # no workspace resolvable


def test_proposal_set_id_pattern():
    pid = new_proposal_set_id()
    assert pid.startswith("propset_") and len(pid.split("_")) == 4


def test_tier2_extras_non_overlap_selection():
    """3a5cb858 shape A: the audition tier keeps only floor candidates that
    overlap NO tier-1 span — an overlapping candidate is the same event
    widened by the lower hysteresis, and tier-1 already carries it."""
    tier1 = [(10.0, 11.0, 0.8), (20.0, 21.5, 0.9)]
    candidates = [(9.8, 11.2, 0.8),    # widened tier-1 event -> dropped
                  (15.0, 15.6, 0.42),  # genuine sub-threshold extra -> kept
                  (21.4, 22.0, 0.45),  # straddles a tier-1 tail -> dropped
                  (30.0, 30.4, 0.38)]  # extra -> kept
    assert tier2_extras(tier1, candidates) == [(15.0, 15.6, 0.42),
                                               (30.0, 30.4, 0.38)]
    # No tier-1 spans: every candidate is an extra; no candidates: empty.
    assert tier2_extras([], candidates) == candidates
    assert tier2_extras(tier1, []) == []


def test_adapter_binding_convention_and_input_errors():
    """DEC 1cfe6d0f: the audio_event_detection adapter binds `Cls(capability_instance)`
    (the duck-typed worker convention) and refuses a non-path training_run with a
    typed input error before touching the inference stack."""
    from cjm_capability_pyannote.propose import PyannoteAudioEventDetectionAdapter
    assert PyannoteAudioEventDetectionAdapter.task_name == "audio_event_detection"
    adapter = PyannoteAudioEventDetectionAdapter(object())
    with pytest.raises(CapabilityInputError):
        adapter.propose("", "audio.wav")
    with pytest.raises(CapabilityInputError):
        adapter.propose(None, "audio.wav")


def test_load_training_run_explicit_workspace(tmp_path, monkeypatch):
    """7839fa98: the explicit workspace reaches the ${WS} fallback — the two-level
    training-runs/<run>/manifest.json layout anchors its primary candidate one
    level too deep, so from a neutral cwd only the caller's workspace resolves
    the checkpoint (env/cwd re-resolution loses it)."""
    from cjm_capability_pyannote.propose import TRAINING_RUN_MANIFEST_FORMAT
    ws_root = tmp_path / "space"
    ws_root.mkdir()
    (ws_root / "cjm-workspace.yaml").write_text("name: t\n")
    run_dir = ws_root / "training-runs" / "run_001"
    run_dir.mkdir(parents=True)
    ckpt = run_dir / "model.ckpt"
    ckpt.write_text("w")
    (run_dir / "manifest.json").write_text(json.dumps({
        "format": TRAINING_RUN_MANIFEST_FORMAT, "run_id": "run_001",
        "classes": ["inhale"],
        "artifact": {"path": "${WS}/training-runs/run_001/model.ckpt"},
    }))
    # Neutral resolution context: no CJM_WORKSPACE, cwd outside any workspace
    monkeypatch.delenv("CJM_WORKSPACE", raising=False)
    neutral = tmp_path / "neutral"
    neutral.mkdir()
    monkeypatch.chdir(neutral)
    with pytest.raises(CapabilityInputError):
        load_training_run(run_dir)  # doubled primary candidate, no fallback
    resolved, checkpoint = load_training_run(run_dir, workspace=Workspace(root=ws_root))
    assert checkpoint == ckpt
    assert resolved["run_id"] == "run_001"
