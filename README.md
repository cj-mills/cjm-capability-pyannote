# cjm-capability-pyannote

<!-- generated from the context graph by `cjm-context-graph readme` — do not edit by hand; edit the graph (the urge to hand-edit = move it on-graph) -->

Speaker-diarization capability backed by pyannote.audio (community-1) — full-source decoded-PCM audio → time-ranged anonymous speaker turns (SPEAKER_00-style cluster labels, never identities), implementing the speaker-diarization adapter interface; pure compute — caching lives in the generic adapter. Default-on rung in the transcription workflow; the correction TUI's assign lane consumes the turns.

## Modules

- **`cjm_capability_pyannote.__init__`**
- **`cjm_capability_pyannote.capability`** — Speaker-diarization tool capability using pyannote.audio (session-D DECs 18d7de80 + d6df3a8e): source audio in, ANONYMOUS time-ranged speaker turns out — the machine half of speaker assignment; identity stays the correction TUI's human lane (DEC 44afb2df).

## API

### `cjm_capability_pyannote.capability`

- `PyannoteConfig` _class_ — Configuration for the pyannote diarization pipeline.
- `PyannoteDiarizationCapability` _class_ — Speaker-diarization tool capability wrapping a pyannote.audio Pipeline.

## Dependencies

**Depends on:** `cjm-capability-primitives`, `cjm-substrate`, `cjm-substrate-torch-utils`, `pyannote.audio`
