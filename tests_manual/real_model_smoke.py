"""Real-model smoke: load the gated community-1 pipeline and diarize one file.

Manual by design (GPU + gated HF model + network on first pull):

    python tests_manual/real_model_smoke.py <audio-path> [num_speakers]

Prints the turn list + metadata; success = typed turns with >=1 cluster."""
import sys
import time

from cjm_capability_pyannote.capability import PyannoteDiarizationCapability


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    audio = sys.argv[1]
    hints = {"num_speakers": int(sys.argv[2])} if len(sys.argv) > 2 else {}
    cap = PyannoteDiarizationCapability()
    cap.initialize()
    t0 = time.monotonic()
    cap.prefetch()
    print(f"pipeline loaded in {time.monotonic() - t0:.1f}s "
          f"(device={cap._loaded_device})")
    t0 = time.monotonic()
    result = cap.diarize(audio, **hints)
    print(f"diarized in {time.monotonic() - t0:.1f}s · {result.metadata}")
    for t in result.turns:
        print(f"  {t.start:8.2f}–{t.end:8.2f}s  {t.speaker}")
    cap.cleanup()
    return 0 if result.turns else 1


if __name__ == "__main__":
    raise SystemExit(main())
