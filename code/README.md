# WillSpeak: code

A snapshot of the working project, November 2026.

| Folder | Contents |
|---|---|
| `src/willspeak/` | The application: the server on the device, audio checks, the recogniser wrapper, the hands-free rules. |
| `scripts/asr/` | Training (`train.py`), the adapter (`block_adapter.py`), export for the Pi (`export.py`), evaluation and intervals (`evaluate.py`, `ci.py`), language-model rescoring (`lm_rescore.py`). |
| `scripts/corpus/` | Fetching TORGO and writing the prompt-disjoint splits (`torgo_manifest.py`). |
| `scripts/matcher/`, `scripts/quality/` | The phrase matcher and the recording-quality gate. |
| `tests/` | Unit and integration tests. |

Left out: voice recordings, model weights (including the two small `.npz` files
under `src/willspeak/models/`), and run logs. The code is here to be read.
