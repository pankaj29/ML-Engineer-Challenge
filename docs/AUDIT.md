# Review Against the Brief

A line-by-line check of the repository against
[CHALLENGE.md](CHALLENGE.md), done on 26 September 2026. Each requirement was
checked against the code and against the running system, not against what the
docs said. This page records what was wrong and what was done about it.

## How it was checked

- The full test suite, plus the 60 tests that need live services, run against
  a real Redis, PostgreSQL, pgvector and the deployed Docker stack.
- The production overlay started for real with generated secrets, and driven
  through the gateway: classify, detect, batch, auth, metrics.
- Every INT8 model run beside its fp32 twin on real images.
- Every number in the docs traced to a report in `benchmarks/reports/`, and
  re-measured where no report existed.

## What was wrong, and the fix

### Production stack

The production overlay had never been started, only validated with
`docker compose config`, and it could not have run.

| Problem | Fix | Verified by |
| --- | --- | --- |
| Redis exited on start: `rename-command "FLUSHALL "` is one argument where Redis needs two | Pass an empty string as the new name | Prod Redis healthy; `FLUSHALL` returns "unknown command" |
| Redis required a password that no client sent, so cache, limiter and Celery all failed | Password in every Redis URL for the API and worker | Prod batch job completed through the passworded broker |
| Worker crashed at start: the production settings check needs `JWT_SECRET`, which the worker was not given | Pass it to the worker | Both workers healthy |
| Prometheus crash-looped on `--web.enable-lifecycle=false`, which it does not accept | Drop the flag; lifecycle is off by default | Prometheus healthy |
| No tables in production: the app only creates them outside production, and nothing ran migrations, so every inference log and batch write failed silently | One-shot `migrate` service runs `alembic upgrade head` before the API and worker | Tables present; inference log and batch rows written |
| Three API replicas each had their own in-memory similarity index | `SIMILARITY_BACKEND=pgvector` in production | Config |
| Port 443 published with no TLS listener, and a mount for certificates that did not exist | Publish 80 only; TLS terminates in front, documented | Config |
| Prometheus scraped `ml-api:8000`, one random replica per scrape, so counters jumped | DNS service discovery: every replica is a target | 3 of 3 replicas `up` in Prometheus |
| Batch metrics were recorded in Celery workers that nothing scraped, so the dashboard's batch panel was always empty | Workers serve multiprocess metrics on 9808; Prometheus scrapes them | Config and unit tests |

### API and security

| Problem | Fix |
| --- | --- |
| `/api/v1/models/health` and `/models/metrics` were public and skipped rate limiting: the public-path check matched any path ending in `/health` or `/metrics`, and `/models/{name}` takes any name | Exact match on the public paths; regression tests |
| Any key could read or cancel any other key's batch job given its id, and cancel revoked ids that never existed | Ownership check, 404 for someone else's job; tests fail on the old code |
| Production accepted the dev JWT secret and the Kubernetes placeholder, both long enough to pass the length check, and the `dev-key-*` API keys | Placeholder values refused at startup; first tests for this validator |
| An inbound `X-Correlation-ID` of any length or content went into logs and a 64-character DB column, where a long one made the log write fail | 1 to 64 safe characters, otherwise replaced |
| After one Redis error the rate limiter stayed on per-process buckets until restart, multiplying the limit by the replica count | Background reconnect every 10 s |
| Batch job status went stale: the worker never updated its row, and after Celery's 24 h expiry a finished job read `pending` again | Worker records running and finished state; status falls back to the row |
| The API enqueued a batch before inserting its row, so a fast worker's update matched nothing | Row first, then enqueue; row marked failed if the enqueue fails |
| The `InferenceFailures` alert could never fire: failures were never counted | Failed and timed-out inferences are counted |
| The worker downloaded `image_url` items with no size cap | Streamed with the same cap as the API |

### Models and optimisation

| Problem | Fix |
| --- | --- |
| **INT8 YOLOv8n detected nothing.** One int8 scale covered box coordinates and class scores, so every score rounded to zero | Quantize convolutions only; calibrate on COCO with the detector's preprocessing. 0.388 mAP50-95 against fp32's 0.392 |
| INT8 results depended on the CPU: signed int8 saturates on x86 without VNNI, so CI's AMD runners got different answers from the same file | Unsigned int8 (U8U8), whose kernel cannot saturate; chosen over 7-bit weights by measurement (`int8_recipes.json`) |
| MinMax calibration let rare outliers set every range; the fine-tuned model's INT8 build lost about 9 points and misread real photos | Percentile calibration for every model |
| The quantizer calibrated every model with classification preprocessing, whatever it was told | Each model's registered preset |
| The quantizer reported 100% agreement for any non-classifier output, and scored agreement on its calibration images | Detector-aware comparison on held-out images |
| The benchmark had no PyTorch baseline, though the brief asks for every format | Eager PyTorch timed in the same interleaved run as ONNX fp32 and INT8: ONNX fp32 is 1.9x to 2.8x faster |
| The TensorRT builder assumed 224x224 for dynamic inputs (YOLO serves 640x640), verified every model with ImageNet preprocessing, and its CLI overwrote the report | Input size and preset passed through; report entries merged by model and precision |
| No test compared INT8 with fp32 through the serving path | `tests/integration/test_quantized_fidelity.py`, strict in CI; fails on the old model |
| The benchmark's speed-up table was empty in every report (`_int8_static` never matched its baseline) | Strip the whole INT8 suffix; test with real names |
| Benchmarks timed each model in one block, and on this hybrid laptop CPU two identical backbones measured 36 and 61 ms | Interleaved rounds; the method is recorded in the report |
| The A/B report compared a model with itself; the drift report ran no tests | A/B supports `name:version@runtime`; drift compares image directories. Both re-run on real data |
| Detector accuracy was cited, never measured | `models/validation/coco_eval.py`: pycocotools mAP through the serving path |
| `quantization_accuracy.json` had no generating script and used the wrong preprocessing | Removed; the full-validation A/B test replaces it |
| `BENCHMARKS_GPU.md` held CPU-fallback numbers under a GPU name, contradicting the fallback report | Removed |

### CI and code quality

| Problem | Fix |
| --- | --- |
| mypy reported 18 errors, hidden by `continue-on-error`, while the README said "mypy clean" | Errors fixed; mypy is a hard gate, run with the project's dependencies installed |
| The `CI passed` gate ignored the security job's result | Gated |
| `release.yml` could not upload its SARIF scan (missing `security-events: write`) | Permission added |

### Documentation

Every number was re-derived from a report, and several had no source.
Figures without evidence (a TensorRT "four runs" range, verification
percentages that matched nothing recorded, resolution experiments whose logs
were not kept) were removed. Figures that could be measured on this machine
were measured, and the scripts are committed: CPU training speed, the INT8
calibration comparison, INT8 agreement for every model, similarity search at
several index sizes, and export parity for the embedding and detection models.
Scaling commands that fail on the dev compose file, a queue-depth command
that read the wrong Redis database, a backup row for a file the app never
writes, and stale paths in comments were corrected. The docs were also cut to
about half their length.

## What remains

- ImageNet accuracy for the default classifier is cited (the validation set
  needs an account).
- TensorRT engines exist for the fine-tuned classifier only. The builder now
  handles each model's input size and preprocessing, and section 8b of the
  Colab notebook builds the other three; running it needs a GPU session.
- API keys carry full access including model reload, by design; scoped JWTs
  are the way to narrow it.
- Confidence is not calibrated.
