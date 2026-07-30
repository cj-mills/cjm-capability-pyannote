"""Tests for the segmentation-finetune adapter — manifest shape, class
selection, spine-file building (windows, real-only holdout, negative
semantics), config plumbing, and typed errors (model-free: no training run;
the real recipe exercises via the dev runner off a real dataset)."""
import json

import pytest

from cjm_capability_pyannote.finetune import (_build_spine_files, _DatasetProtocol, _eligible_spines,
                                              FinetuneConfig, load_dataset, new_training_run_id,
                                              PyannoteSegmentationFinetuneAdapter, run_finetune,
                                              select_classes, TrainingRunManifest)
from cjm_substrate.core.errors import CapabilityInputError
from cjm_substrate.core.workspace import Workspace

SKELETON = "sha256:" + "ab" * 32
SOURCE = "11111111-2222-3333-4444-555555555555"


def _event(start, end, label="inhale", tag="real"):
    return {
        "kind": "labeled_span", "source_id": SOURCE, "source_title": "t",
        "source_path": "/media/a.mp3", "source_content_hash": "sha256:" + "cd" * 32,
        "skeleton_hash": SKELETON, "insert_id": "x", "label": label, "text": "",
        "speech": False, "start_time": start, "end_time": end, "split": "train",
        "provenance": {"tag": tag, "sessions": [], "op_ids": []},
    }


def _region(start, end, kind="speech"):
    return {"kind": kind, "source_id": SOURCE, "skeleton_hash": SKELETON,
            "start_time": start, "end_time": end}


@pytest.fixture
def dataset_dir(tmp_path):
    """A tiny on-disk dataset in the leg-2 layout (manifest + jsonl files)."""
    d = tmp_path / "datasets" / "dataset_test"
    d.mkdir(parents=True)
    manifest = {
        "format": "cjm-transcript-correction-core/dataset-manifest",
        "version": "0.1.0",
        "dataset_id": "dataset_test",
        "created_at": 0.0,
        "config": {},
        "graph_db_path": "db",
        "journals": [],
        "session_purpose_policy": {},
        "split_policy": {},
        "augmentation_policy": {},
        "class_vocabulary": {"inhale": 3, "empty": 2, "echo": 1},
        "spines": [
            {"source_id": SOURCE, "skeleton_hash": SKELETON, "extraction_status": "in_progress",
             "annotated_through": 100.0, "eligible": True, "examples": 3, "skipped": {}},
            {"source_id": SOURCE, "skeleton_hash": None, "extraction_status": "in_progress",
             "annotated_through": None, "eligible": False, "examples": 0, "skipped": {}},
        ],
        "files": {"events": "events.jsonl", "regions": "regions.jsonl"},
        "counts": {"examples": 3},
    }
    events = [
        _event(10.0, 10.5),                      # train window
        _event(50.0, 50.4),                      # train window
        _event(95.0, 95.3),                      # dev window (real)
        _event(96.0, 96.2, tag="synthetic"),     # dev window — real-only filter drops it
        _event(20.0, 20.1, label="empty"),       # excluded label: never a class
    ]
    regions = [_region(0.0, 10.0), _region(10.5, 50.0), _region(50.4, 99.0),
               _region(99.0, 100.0, kind="negative")]
    (d / "manifest.json").write_text(json.dumps(manifest))
    (d / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    (d / "regions.jsonl").write_text("\n".join(json.dumps(r) for r in regions))
    return d


@pytest.fixture
def adapter():
    a = PyannoteSegmentationFinetuneAdapter()
    a.initialize()
    return a


def test_config_defaults_and_schema(adapter):
    cfg = adapter.get_current_config()
    assert cfg["base_model_id"] == "pyannote/segmentation-3.0"
    assert cfg["holdout_fraction"] == 0.1
    assert cfg["exclude_labels"] == ["empty"]
    schema = adapter.get_config_schema()
    assert "base_model_id" in schema["properties"] and "duration" in schema["properties"]


def test_training_run_id_pattern():
    rid = new_training_run_id()
    assert rid.startswith("trainrun_") and len(rid.split("_")) == 4


def test_manifest_shape_and_ws_tokens(tmp_path):
    """The manifest pattern contract: format tag + version + consumed pointer,
    workspace-owned paths recorded in ${WS} token form."""
    ws = Workspace(root=tmp_path)
    m = TrainingRunManifest(
        run_id="trainrun_x", created_at=1.0, config={"seed": 42},
        dataset_manifest=str(tmp_path / "datasets" / "d" / "manifest.json"),
        dataset_id="d",
        artifact={"path": str(tmp_path / "training-runs" / "trainrun_x" / "model.ckpt"),
                  "content_hash": "sha256:00", "size_bytes": 1},
    )
    out = m.save(tmp_path / "training-runs" / "trainrun_x" / "manifest.json", workspace=ws)
    data = json.loads(out.read_text())
    assert data["format"] == "cjm-capability-pyannote/training-run-manifest"
    assert data["version"] == "0.1.0"
    assert data["dataset_manifest"] == "${WS}/datasets/d/manifest.json"
    assert data["artifact"]["path"] == "${WS}/training-runs/trainrun_x/model.ckpt"


def test_load_dataset_rejects_wrong_format(tmp_path):
    bad = tmp_path / "manifest.json"
    bad.write_text(json.dumps({"format": "something-else"}))
    with pytest.raises(CapabilityInputError):
        load_dataset(bad)


def test_load_dataset_reads_files(dataset_dir):
    manifest, events, regions = load_dataset(dataset_dir / "manifest.json")
    assert manifest["dataset_id"] == "dataset_test"
    assert len(events) == 5 and len(regions) == 4
    assert len(_eligible_spines(manifest)) == 1


def test_select_classes_thresholds_excludes_and_speech(dataset_dir):
    manifest, _, _ = load_dataset(dataset_dir / "manifest.json")
    cfg = FinetuneConfig(min_class_count=2)
    # echo (1) under threshold, empty excluded, speech appended last
    assert select_classes(manifest, cfg) == ["inhale", "speech"]
    cfg = FinetuneConfig(min_class_count=2, include_speech_class=False)
    assert select_classes(manifest, cfg) == ["inhale"]
    with pytest.raises(CapabilityInputError):
        select_classes(manifest, FinetuneConfig(min_class_count=10, include_speech_class=False))


def test_build_spine_files_windows_and_real_only(dataset_dir, tmp_path):
    """The split contract: annotated head splits by time at watermark*(1-f);
    the dev annotation is REAL-ONLY; speech regions crop into their window;
    everything above the watermark never appears."""
    manifest, events, regions = load_dataset(dataset_dir / "manifest.json")
    cfg = FinetuneConfig(min_class_count=2)
    classes = select_classes(manifest, cfg)
    spine = _eligible_spines(manifest)[0]
    audio = tmp_path / "a.wav"
    train_file, dev_file, counts = _build_spine_files(spine, events, regions, classes, cfg, audio)

    assert list(train_file["annotated"])[0].end == pytest.approx(90.0)
    assert list(dev_file["annotated"])[0] .start == pytest.approx(90.0)
    assert train_file["classes"] == classes and dev_file["classes"] == classes
    assert train_file["scope"] == "file"

    # train: 2 inhales + speech regions cropped to [0, 90); the 'empty' span is no class
    assert counts["train"]["inhale"] == 2
    # dev: ONE real inhale — the synthetic-tagged event is filtered out
    assert counts["dev"]["inhale"] == 1
    dev_labels = [l for _s, _t, l in dev_file["annotation"].itertracks(yield_label=True)]
    assert dev_labels.count("inhale") == 1
    # speech region [50.4, 99.0] appears cropped in BOTH windows
    assert counts["train"]["speech"] == 3 and counts["dev"]["speech"] == 1
    # nothing beyond the watermark
    for f in (train_file, dev_file):
        for seg in f["annotation"].itertracks():
            assert seg[0].end <= 100.0

    proto = _DatasetProtocol([train_file], [dev_file])
    assert [f["uri"] for f in proto.train()] == [train_file["uri"]]
    assert [f["uri"] for f in proto.development()] == [dev_file["uri"]]


def test_run_finetune_validates_holdout_and_workspace(dataset_dir, tmp_path, monkeypatch):
    monkeypatch.delenv("CJM_WORKSPACE", raising=False)
    monkeypatch.chdir(tmp_path)  # no workspace marker anywhere above tmp_path
    with pytest.raises(CapabilityInputError):
        run_finetune(dataset_dir / "manifest.json", config={"holdout_fraction": 0.0})
    with pytest.raises(CapabilityInputError):
        run_finetune(dataset_dir / "manifest.json")  # no workspace resolvable


def test_adapter_input_errors(adapter):
    with pytest.raises(CapabilityInputError):
        adapter.finetune("")
    with pytest.raises(CapabilityInputError):
        adapter.finetune(123)
    with pytest.raises(CapabilityInputError):
        adapter.finetune("/nonexistent/manifest.json")
