# cjm-capability-pyannote

A [pyannote.audio](https://github.com/pyannote/pyannote-audio) speaker-diarization
capability for the [cjm-substrate](https://github.com/cj-mills/cjm-substrate)
runtime: source audio in, **anonymous** time-ranged speaker turns out
(`SpeakerDiarizationResult` — wire-registered, typed across the worker boundary).

- Default pipeline: [`pyannote/speaker-diarization-community-1`](https://hf.co/pyannote/speaker-diarization-community-1)
  (**HF-gated**: accept the model's user conditions and provision a token in the
  worker env — `hf auth login` or `HF_TOKEN`; the model downloads into the
  standard HF cache on first load).
- Task channel: `task=speaker_diarization` via
  [cjm-speaker-diarization-adapter-interface](https://github.com/cj-mills/cjm-speaker-diarization-adapter-interface)
  (surface match: `diarize` + `get_current_config`).
- Speaker-count hints are **per-call** knowledge, not config:
  `diarize(audio, num_speakers=2)` / `min_speakers` / `max_speakers`.
- Turn times index into the caller's original audio; overlapping turns are
  signal (overlapped speech). Cluster labels (`SPEAKER_00`, …) are
  result-scoped — mapping them to named identities is downstream human
  judgment (the correction TUI's assignment lane), never this capability's job.

## Install (as a substrate capability)

Add an entry to your workspace `capabilities.yaml` (env file + package +
`cjm-capability-primitives` interface lib + the generic diarization adapter),
then:

```bash
cjm-ctl install-all --capabilities capabilities.yaml
```
