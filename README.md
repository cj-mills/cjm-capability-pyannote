# cjm-capability-pyannote

<!-- generated from the context graph by `cjm-context-graph readme` — do not edit by hand; edit the graph (the urge to hand-edit = move it on-graph) -->

Speaker-diarization capability backed by pyannote.audio (community-1) — full-source decoded-PCM audio → time-ranged anonymous speaker turns (SPEAKER_00-style cluster labels, never identities), implementing the speaker-diarization adapter interface; pure compute — caching lives in the generic adapter. Default-on rung in the transcription workflow; the correction TUI's assign lane consumes the turns.

## Modules

- **`cjm_capability_pyannote.__init__`**
- **`cjm_capability_pyannote.capability`** — Speaker-diarization tool capability using pyannote.audio (session-D DECs 18d7de80 + d6df3a8e): source audio in, ANONYMOUS time-ranged speaker turns out — the machine half of speaker assignment; identity stays the correction TUI's human lane (DEC 44afb2df).
- **`cjm_capability_pyannote.finetune`** — Segmentation-finetune task adapter (flywheel leg 3): a DatasetManifest in, a
- **`cjm_capability_pyannote.propose`** — Event-proposal generation (flywheel leg 4): a TrainingRunManifest in, a

## API

### `cjm_capability_pyannote.capability`

- `PyannoteConfig` _class_ — Configuration for the pyannote diarization pipeline.
- `PyannoteDiarizationCapability` _class_ — Speaker-diarization tool capability wrapping a pyannote.audio Pipeline.

### `cjm_capability_pyannote.finetune`

- `FinetuneConfig` _class_ — Configuration for one segmentation finetune run.
- `PyannoteSegmentationFinetuneAdapter` _class_ — Segmentation-finetune task adapter wrapping the pyannote training recipe.
- `TrainingRunManifest` _class_ — Durable record of one finetune run (chainable; DEC 16159e09).
- `encounter_class_order` _function_ — The class order upstream target-building actually uses.
- `load_dataset` _function_ — Load a DatasetManifest and its data files.
- `main` _function_ — Dev runner: `python -m cjm_capability_pyannote.finetune <manifest> [...]`.
- `new_training_run_id` _function_ — Generate a unique, sortable training-run id (the new_run_id pattern, trainrun kind).
- `run_finetune` _function_ — Run one segmentation finetune off a DatasetManifest.
- `select_classes` _function_ — Pick the trained classes from the dataset's observed OPEN vocabulary.

### `cjm_capability_pyannote.propose`

- `ProposalSetManifest` _class_ — Durable record of one proposal-generation run (chainable; DEC 16159e09).
- `ProposeConfig` _class_ — Configuration for one proposal-generation run.
- `PyannoteAudioEventDetectionAdapter` _class_ — audio_event_detection task adapter over `run_propose` (DEC 1cfe6d0f).
- `load_training_run` _function_ — Load a TrainingRunManifest and resolve its checkpoint artifact.
- `main` _function_ — Dev runner: `python -m cjm_capability_pyannote.propose <training-run> <audio> [...]`.
- `new_proposal_set_id` _function_ — Generate a unique, sortable proposal-set id (the new_run_id pattern, propset kind).
- `run_propose` _function_ — Generate event-span proposals over a source window.
- `spans_from_scores` _function_ — Extract event spans from frame scores by onset/offset hysteresis.
- `tier2_extras` _function_ — Select the audition tier: floor-extraction spans that do not overlap any

## Dependencies

**Depends on:** `cjm-capability-primitives`, `cjm-substrate`, `cjm-substrate-torch-utils`, `pyannote.audio`
