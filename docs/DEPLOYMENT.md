# Deployment and Scaling Guide

How to run this system for real, and what to do when it misbehaves.

---

## Contents

1. [Local development](#1-local-development)
2. [Production deployment](#2-production-deployment)
3. [Scaling](#3-scaling)
4. [Monitoring](#4-monitoring)
5. [Shipping a new model](#5-shipping-a-new-model)
6. [Troubleshooting](#6-troubleshooting)
7. [Backup and recovery](#7-backup-and-recovery)

---

## 1. Local development

### Prerequisites

* Docker with Compose v2
* Python 3.11+ (only to prepare the models; the services run in containers)
* ~4 GB free disk for model artifacts

### First run

```bash
# 1. Configuration
cp .env.example .env

# 2. Download and export the models (one-off, ~5 minutes)
pip install -r requirements-train.txt onnxscript
python scripts/prepare_models.py

# 3. Start everything
docker compose up -d

# 4. Confirm
curl http://localhost/api/v1/health
```

All seven services should report healthy within about 90 seconds (the API's
`start_period` allows for loading three ONNX models):

```bash
docker compose ps
```

```
NAME              STATUS
mlcv-api          Up (healthy)
mlcv-gateway      Up (healthy)
mlcv-grafana      Up (healthy)
mlcv-postgres     Up (healthy)
mlcv-prometheus   Up (healthy)
mlcv-redis        Up (healthy)
mlcv-worker       Up (healthy)
```

### Where things are

| Service | URL | Credentials |
| --- | --- | --- |
| API (via gateway) | http://localhost | `X-API-Key: dev-key-pro` |
| API (direct, dev only) | http://localhost:8000 | same |
| Swagger UI | http://localhost:8000/docs | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | `admin` / `admin` |

### Everyday commands

```bash
docker compose logs -f ml-api        # follow one service
docker compose restart ml-api        # restart after a config change
docker compose up -d --build ml-api  # rebuild after a code change
docker compose down                  # stop, keep data
docker compose down -v               # stop and DELETE all volumes
```

---

## 2. Production deployment

### Before you start

Production is **not** `docker compose up`. The overlay changes five things
that matter:

1. **No direct ports.** Only the gateway is reachable. Postgres and Redis are
   not exposed to the host at all.
2. **Replicas.** 3 API and 2 worker containers by default.
3. **Secrets are required.** No defaults — the stack refuses to start without
   them.
4. **Read-only root filesystems**, with explicit tmpfs for scratch space.
5. **Rolling updates** with automatic rollback.

### Generate secrets

```bash
python -c "import secrets; print('JWT_SECRET=' + secrets.token_urlsafe(48))"
python -c "import secrets; print('POSTGRES_PASSWORD=' + secrets.token_urlsafe(32))"
python -c "import secrets; print('REDIS_PASSWORD=' + secrets.token_urlsafe(32))"
python -c "import secrets; print('GRAFANA_PASSWORD=' + secrets.token_urlsafe(24))"
python -c "import secrets; print('API_KEYS=' + secrets.token_urlsafe(32) + ':pro')"
```

Put them in the deployment environment — a secrets manager, not a file in the
repository.

> The stack **verifies** this. Starting the production overlay without
> `JWT_SECRET` fails immediately with
> `required variable JWT_SECRET is missing a value`. That is deliberate: a
> missing secret must be a loud startup failure, never a silent default.

### Deploy

```bash
export ENVIRONMENT=production
export JWT_SECRET=...  API_KEYS=...  POSTGRES_PASSWORD=...
export REDIS_PASSWORD=...  GRAFANA_PASSWORD=...

docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Production checklist

* [ ] All secrets set from a secrets manager, none in git
* [ ] `ENVIRONMENT=production` (the app refuses to start with `DEBUG=true` or
      `AUTH_ENABLED=false` in this mode)
* [ ] TLS terminated at the gateway or a load balancer in front
* [ ] `CORS_ORIGINS` set to your actual origins, not `*`
* [ ] Model artifacts on a persistent, backed-up volume
* [ ] Postgres backups scheduled (see §7)
* [ ] Prometheus retention and disk sized
* [ ] Alert routing configured in Alertmanager
* [ ] Log shipping configured (logs are JSON on stdout)
* [ ] Resource limits matched to the host
* [ ] Regression baselines recorded on production-like hardware

### TLS

Nginx is ready for it; mount certificates at `docker/nginx/certs` and add a
443 server block, or terminate TLS at a cloud load balancer and leave the
gateway on HTTP inside the private network. The second is usually simpler.

---

## 3. Scaling

### Scale the API

```bash
docker compose up -d --scale ml-api=5
```

Nginx discovers the replicas through Docker's DNS. No configuration change.

**When:** `inference_in_progress` regularly at the concurrency limit, or p95
latency rising while inference latency stays flat (meaning time is spent
queueing).

> Scale out rather than raising `MAX_CONCURRENT_INFERENCES`. Inference is
> CPU-bound: a higher limit on the same cores makes every request slower
> without increasing throughput.

### Scale workers

```bash
docker compose up -d --scale worker=4
```

**When:** batch jobs sit in `pending` for longer than users tolerate. Workers
scale on queue depth, independently of API traffic.

### Sizing guide

Measured: ~38.7 req/s end-to-end through the full stack on a 22-core CPU host.

| Target | API replicas | Workers | Notes |
| --- | ---: | ---: | --- |
| < 30 req/s | 1 | 1 | Development or light production |
| 30-100 req/s | 3 | 2 | The production default |
| 100-500 req/s | 8-10 | 4 | Raise Redis memory; consider read replicas |
| > 500 req/s | — | — | Move to GPU inference; the CPU-first assumptions no longer hold |

### Vertical tuning

| Variable | Guidance |
| --- | --- |
| `MAX_CONCURRENT_INFERENCES` | ~1-2 per available CPU core |
| `API_CPU_LIMIT` | At least 2.0; ONNX Runtime benefits from more |
| `API_MEMORY_LIMIT` | ~1 GB base + ~200 MB per loaded model |
| `REDIS_MAXMEMORY` | Larger = better hit rate. `allkeys-lru` eviction is already set. |

> **Thread counts are pinned to 1** in both images (`OMP_NUM_THREADS=1` etc.)
> and concurrency is handled by running more containers. Left unset, every
> numerical library spawns a thread per *host* core — inside a container
> limited to 2 CPUs that means dozens of threads fighting over 2 cores, which
> is slower than a single thread.

---

## 4. Monitoring

### The four numbers that matter

Open Grafana → *ML CV API — Overview*. The top row answers "is it healthy?":

| Panel | Healthy | Investigate |
| --- | --- | --- |
| Error rate | < 1% | > 5% sustained |
| p95 inference latency | < 500 ms | > 1 s |
| Cache hit rate | > 50% | < 10% |
| Models loaded | 3 | 0 is critical |

### Alerts

Eleven rules in `monitoring/prometheus/alerts.yml`, split by severity:

* **critical** — `MLAPIDown`, `NoModelsLoaded`, `HighServerErrorRate`
* **warning** — `SlowInference`, `InferenceFailures`, `InferenceQueueSaturated`,
  `CacheHitRateCollapsed`, `ModelLoadFailures`, `SlowRequests`,
  `ElevatedClientErrors`
* **info** — `HighRateLimitRejections`

Every rule alerts on a **symptom** (users are affected) rather than a cause
(CPU is busy), and every one carries a description saying what to do. Rules
that only produce "huh, weird" are noise, and noise trains people to ignore
the alerts that matter.

### Logs

JSON on stdout, one object per line. Ship them with whatever you already use.

```bash
# Everything for one request, across API and worker
docker compose logs | grep "0f6c1d8a"

# Errors only
docker compose logs ml-api | grep '"level": "ERROR"'

# Slow requests
docker compose logs ml-api | grep '"slow": true'
```

The correlation id is the thread that ties together the gateway access log,
every API log line, the worker, and the row in Postgres.

---

## 5. Shipping a new model

Model artifacts are mounted as a volume, **not** baked into the image, so new
weights do not require rebuilding and redeploying the service.

```bash
# 1. Export and register the new version
python -m models.optimization.export_onnx --model resnet50 \
       --checkpoint path/to/finetuned.pt --num-classes 200

python -m models.registry register \
       --name resnet50-v2 --version 2.0.0 --task classification \
       --onnx resnet50-v2.onnx --labels labels.json

# 2. Validate it BEFORE it serves traffic
python -m models.validation.validate --model resnet50-v2:2.0.0

# 3. Check it is not a regression
python -m models.validation.regression check --model resnet50-v2:2.0.0

# 4. Compare against the incumbent
python -m models.validation.ab_test \
       --champion resnet50:1.0.0 --challenger resnet50-v2:2.0.0 --samples 500

# 5. Load it without a restart
curl -X POST http://localhost/api/v1/models/reload \
     -H "Authorization: Bearer <admin-token>"
```

Step 4 is the one people skip. It runs a **paired McNemar test**, which asks
whether the difference is real or luck, and reports a confidence interval on
the accuracy delta alongside the latency change. It will tell you not to
promote a model that is 0.2% more accurate and 300 ms slower.

### Canary rollout

`models/validation/ab_test.py` provides deterministic hash-based traffic
splitting: send 10% of users to the challenger, keep the rest on the champion,
and compare real outcomes. Assignment is by hashed user id, so a user stays on
one variant — random per-request assignment would both ruin the statistics and
produce visibly inconsistent behaviour.

### Rollback

```bash
# Per request, immediately
{"model_version": "1.0.0"}

# Or globally: re-flag the default and reload
python -m models.registry retire --name resnet50-v2 --version 2.0.0
curl -X POST http://localhost/api/v1/models/reload -H "Authorization: Bearer <admin>"
```

`/models/reload` also **invalidates cached results** for removed models, so
predictions from retired weights cannot keep being served from Redis.

---

## 6. Troubleshooting

### The API container is unhealthy

```bash
docker logs mlcv-api --tail 50
```

| Symptom | Cause | Fix |
| --- | --- | --- |
| `registry_missing` | Models not prepared | `python scripts/prepare_models.py` |
| `model_load_failed` | Artifact missing from the volume | `python -m models.registry validate` |
| Unhealthy during startup | Still loading models | Wait — `start_period` is 90 s |
| `JWT_SECRET must be set` | Production without secrets | Working as designed; set them |

### Every request returns 401

`AUTH_ENABLED=true` with no `API_KEYS` configured. The service **fails
closed** on purpose — there is no default credential. Set `API_KEYS`.

### Every request returns 429

Check your tier's allowance:

```bash
curl -sD- -o/dev/null -H "X-API-Key: your-key" http://localhost/api/v1/models | grep -i ratelimit
```

Two limits exist: the per-tier one in the API, and a coarser per-IP one at the
gateway (30 r/s on inference paths). The response body tells you which.

### Latency has risen

Work through it in this order:

1. **Grafana → "HTTP vs inference latency".** If the gap widened but inference
   is flat, the model is fine and the time is going on queueing, validation or
   the database.
2. **Cache hit rate.** A collapse means every request is recomputing.
3. **`inference_in_progress`.** At the ceiling → scale out.
4. **Upload size distribution.** A jump in image sizes explains a latency rise
   with no code change at all.

### Batch jobs stay `pending`

```bash
docker compose ps worker
docker compose logs worker --tail 50
docker compose exec redis redis-cli LLEN inference   # queue depth
```

Worker down, or the queue is deeper than the workers can drain. Scale workers.

### Redis is down

The system keeps working: cache misses, and per-process rate limiting. You
will see `cache_unavailable` and `rate_limiter_degraded_to_local_buckets` in
the logs, and `/health` reports `degraded` while still returning 200. Fix
Redis at normal urgency, not emergency urgency.

### Postgres is down

Inference logging pauses. **Predictions are unaffected.** Drift detection and
audit history have a gap for the outage window.

---

## 7. Backup and recovery

### What needs backing up

| Data | Where | Priority |
| --- | --- | --- |
| Model artifacts | `models/artifacts/` volume | **High** — regenerable, but slowly |
| Model registry | `models/registry.json` | **High** — in git; keep it there |
| Inference logs | `postgres_data` volume | Medium — audit and drift history |
| Similarity index | `models/artifacts/similarity_index.npz` | Medium — re-embeddable |
| Redis | `redis_data` volume | Low — cache is disposable; the queue is not |
| Grafana | `grafana_data` volume | Low — dashboards are provisioned from git |

### Postgres backup

```bash
docker compose exec -T postgres pg_dump -U mluser mldb | gzip > backup-$(date +%F).sql.gz

# Restore
gunzip -c backup-2026-09-22.sql.gz | docker compose exec -T postgres psql -U mluser mldb
```

### Full recovery from nothing

```bash
git clone <repo> && cd ml-engineer-challenge
cp .env.example .env            # then set real secrets
python scripts/prepare_models.py
docker compose up -d
gunzip -c backup.sql.gz | docker compose exec -T postgres psql -U mluser mldb
```

The deliberate property here: **the system can be rebuilt from git plus one
command.** Model artifacts are regenerated by `prepare_models.py` rather than
being irreplaceable binaries, and every dashboard, alert rule and data source
is provisioned from files in the repository.
