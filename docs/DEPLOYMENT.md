# Deployment and Scaling

How to run the system, scale it, and fix it when it misbehaves.

1. [Local development](#1-local-development)
2. [Production with Compose](#2-production-with-compose)
3. [Scaling](#3-scaling)
4. [Kubernetes](#4-kubernetes)
5. [Monitoring](#5-monitoring)
6. [Shipping a new model](#6-shipping-a-new-model)
7. [Troubleshooting](#7-troubleshooting)
8. [Backup and recovery](#8-backup-and-recovery)

---

## 1. Local development

You need Docker with Compose v2, Python 3.11 or 3.12 for the model tooling,
and Git LFS for the model files.

```bash
git lfs pull                 # the committed models, about 500 MB
cp .env.example .env
docker compose up -d
curl http://localhost/api/v1/health
```

All seven services report healthy within about 90 seconds; the API's
`start_period` allows for loading three models. `docker compose ps` shows them.

| Service | URL | Credentials |
| --- | --- | --- |
| API through the gateway | http://localhost | `X-API-Key: dev-key-pro` |
| API direct (dev only) | http://localhost:8000 | same |
| Swagger UI | http://localhost:8000/docs | |
| Prometheus | http://localhost:9090 | |
| Grafana | http://localhost:3000 | `admin` / `admin` |

```bash
docker compose logs -f ml-api        # follow one service
docker compose up -d --build ml-api  # rebuild after a code change
docker compose down                  # stop, keep data
docker compose down -v               # stop and delete all volumes
```

To re-export the models from scratch instead of using the committed ones:
`pip install -r requirements-train.txt onnxscript`, then
`python scripts/prepare_models.py`. The fine-tuned classifier cannot be
re-created this way; it needs a GPU training run.

---

## 2. Production with Compose

The production file is an overlay on the base file, never used alone:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

What it changes:

- Only the gateway publishes a port (80). Postgres, Redis, the API,
  Prometheus and Grafana are unreachable from outside the host.
- Three API replicas and two workers, rolling updates with rollback.
- Secrets are required. `${VAR:?}` makes Compose refuse to start without
  them, and the app itself refuses the development defaults and the
  Kubernetes placeholders even when they are long enough.
- Redis requires a password, and FLUSHALL, FLUSHDB and CONFIG are disabled.
  Every client URL carries the password.
- A one-shot `migrate` service runs `alembic upgrade head` before the API and
  worker start. In production the app does not create tables itself.
- The similarity index is pgvector, so all replicas share one index.
- Read-only root filesystems with tmpfs scratch space, and
  `no-new-privileges` on every service.

### Secrets

```bash
python -c "import secrets; print('JWT_SECRET=' + secrets.token_urlsafe(48))"
python -c "import secrets; print('POSTGRES_PASSWORD=' + secrets.token_urlsafe(32))"
python -c "import secrets; print('REDIS_PASSWORD=' + secrets.token_urlsafe(32))"
python -c "import secrets; print('GRAFANA_PASSWORD=' + secrets.token_urlsafe(24))"
python -c "import secrets; print('API_KEYS=' + secrets.token_urlsafe(32) + ':pro')"
```

Use `token_urlsafe` for the Redis password: it goes inside a URL. Keep the
values in your deployment environment or a secrets manager, not in the
repository.

### Checklist

- [ ] Secrets set from a secrets manager
- [ ] TLS terminated in front of the host (see below)
- [ ] `CORS_ORIGINS` set to your real origins, not `*`
- [ ] Postgres backups scheduled (section 8)
- [ ] Alertmanager routing configured; the rules exist, the destination does not
- [ ] Regression baselines recorded on the production hardware

### TLS

The gateway listens on port 80 only. Terminate TLS at a load balancer in front
of the host, which is the simpler setup. To terminate in nginx instead, add a
`listen 443 ssl` server block to `docker/nginx/nginx.conf`, mount the
certificates, and publish 443 in the overlay.

### Dev and prod on one machine

Both files use the project name `mlcv`, so they share volumes. Running the
test suite against the dev stack leaves a `FLUSHDB` in Redis's append-only
log, and the production Redis, which disables FLUSHDB, then refuses to replay
it. Use a separate project name for production on a shared host:
`docker compose -p mlcv-prod -f docker-compose.yml -f docker-compose.prod.yml up -d`.

---

## 3. Scaling

Everything here scales one host. For load-driven replica counts or more than
one machine, see section 4.

Scaling only works with the production overlay. The base file gives the API
and worker fixed container names and publishes port 8000, and Compose cannot
run two containers with one name.

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --scale ml-api=5
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --scale worker=4
```

nginx and Prometheus both find replicas through Docker's DNS, so neither needs
a config change.

**Scale the API** when `inference_in_progress` sits at the concurrency limit,
or p95 request latency rises while inference latency stays flat. Raising
`MAX_CONCURRENT_INFERENCES` does not help: inference is CPU-bound, and more of
it on the same cores makes every request slower.

**Scale workers** when batch jobs stay `pending`. Workers follow queue depth,
not API traffic.

### Sizing

Measured end to end through the dev stack on a 22-core laptop: ⟦LOADTEST_RPS⟧
requests per second with 20 concurrent users (`benchmarks/reports/`).

| Target | API replicas | Workers | Notes |
| --- | ---: | ---: | --- |
| under 30 req/s | 1 | 1 | Development or light production |
| 30 to 100 req/s | 3 | 2 | The production default |
| 100 to 500 req/s | 8 to 10 | 4 | More Redis memory |
| over 500 req/s | | | GPU inference; the CPU assumptions stop holding |

Both images pin `OMP_NUM_THREADS=1`. Without it every numerical library starts
a thread per host core, and inside a container limited to two CPUs that is
dozens of threads fighting over two cores.

---

## 4. Kubernetes

The manifests in [`k8s/`](../k8s/README.md) hand the replica count to an HPA
and add what Compose has no concept of: disruption budgets, pod security and
network policy. Move to it when traffic varies through the day, when the API
must survive losing a node, or when one host runs out of CPU.

You need a v1.29+ cluster (verified on v1.33.1 with kind), `metrics-server`,
an nginx ingress controller, a registry the cluster can pull from, and
somewhere to serve model artifacts from (`ARTIFACT_SOURCE`).

```bash
kubectl kustomize k8s/base            # render first; catches most mistakes
kubectl apply -k k8s/base
kubectl -n mlcv rollout status deployment/ml-api
python scripts/smoke_test_api.py --base-url https://your-host --api-key YOUR_KEY
```

| | Compose | Kubernetes |
| --- | --- | --- |
| Replica count | `--scale`, by hand | HPA, 2 to 10 on CPU at 70% |
| Edge | nginx container | Ingress controller |
| Model artifacts | Bind mount | Init container, checksum-verified |
| Schema | `migrate` service (prod) | `alembic upgrade head` init container |
| Similarity index | pgvector (prod) | pgvector |
| Drift checks | On demand | Weekly CronJob |

Postgres is a single StatefulSet in these manifests. That is fine for a demo;
use a managed database or an operator such as CloudNativePG in production.
The full walkthrough, canary releases included, is in
[`k8s/README.md`](../k8s/README.md).

---

## 5. Monitoring

Grafana's *ML CV API Overview* dashboard is provisioned automatically. The top
row answers "is it healthy":

| Panel | Healthy | Investigate |
| --- | --- | --- |
| Error rate | under 1% | over 5% sustained |
| p95 inference latency | under 500 ms | over 1 s |
| Cache hit rate | over 50% | under 10% |
| Models loaded | 3 | 0 is critical |

Prometheus scrapes every API replica and every worker as separate targets.
Batch job counts and durations come from the workers, which serve them on port
9808.

Eleven alert rules live in `monitoring/prometheus/alerts.yml`:

- critical: `MLAPIDown`, `NoModelsLoaded`, `HighServerErrorRate`
- warning: `SlowInference`, `InferenceFailures`, `InferenceQueueSaturated`,
  `CacheHitRateCollapsed`, `ModelLoadFailures`, `SlowRequests`,
  `ElevatedClientErrors`
- info: `HighRateLimitRejections`

Each alerts on a symptom users feel, not on a cause like busy CPU, and each
says what to do.

Logs are JSON on stdout. The correlation id ties the gateway access log, the
API, the worker and the Postgres row together:

```bash
docker compose logs | grep "0f6c1d8a"                      # one request, everywhere
docker compose logs ml-api | grep '"level": "ERROR"'
```

---

## 6. Shipping a new model

```bash
# 1. Export and register the new version
python -m models.optimization.export_onnx --model resnet50 \
       --checkpoint path/to/finetuned.pt --num-classes 200
python -m models.registry register --name resnet50-v2 --version 2.0.0 \
       --task classification --onnx resnet50-v2.onnx --labels labels.json

# 2. Validate before it serves anything
python -m models.validation.validate --model resnet50-v2:2.0.0

# 3. Check it is not a regression
python -m models.validation.regression check --model resnet50-v2:2.0.0

# 4. Compare with the model it would replace
python -m models.validation.ab_test --champion resnet50:1.0.0 \
       --challenger resnet50-v2:2.0.0 --samples 500

# 5. Load it without a restart
curl -X POST http://localhost/api/v1/models/reload -H "X-API-Key: <key>"
```

Step 4 runs a paired McNemar test, so it can tell a real improvement from
luck, and it reports the latency change next to the accuracy change. Append
`@onnx_int8` to either side to compare runtimes of one model; that is how the
INT8 decision in [TECHNICAL.md](TECHNICAL.md) was made.

Register a new version rather than overwriting weights. Versions are how
callers pin a model and how a rollback names its target. If weights are
overwritten anyway, the cache copes: its keys include a hash of the artifact,
so stale entries are never read again.

Rollback is per request (`"model_version": "1.0.0"`) or global:

```bash
python -m models.registry retire --name resnet50-v2 --version 2.0.0
curl -X POST http://localhost/api/v1/models/reload -H "X-API-Key: <key>"
```

A reload also invalidates cached results for removed models.

---

## 7. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| API unhealthy for the first 90 s | Still loading models. Wait. |
| `registry_missing` or `model_load_failed` | Artifacts missing or LFS pointers. `git lfs pull`, then `python -m models.registry validate`. |
| `JWT_SECRET must be set` or `is a placeholder value` | Production without real secrets. Working as designed. |
| Every request 401 | `API_KEYS` is empty. Auth fails closed; there is no default key. |
| Every request 429 | Check `X-RateLimit-*` headers. The per-IP gateway limit (30 r/s on inference paths) is separate from the per-tier API limit. |
| Batch jobs stay `pending` | Worker down or queue too deep. `docker compose exec redis redis-cli -n 1 LLEN inference` shows the depth (in production add `-a $REDIS_PASSWORD`). |
| Similarity search misses an indexed image | More than one replica with `SIMILARITY_BACKEND=memory`. Use pgvector. |

When latency rises, check in this order: the "HTTP vs inference latency"
panel (a wider gap with flat inference means queueing, not the model), the
cache hit rate, `inference_in_progress` against its limit, then the upload
size distribution.

If Redis goes down, the API keeps serving: cache misses, and per-process rate
limits until the limiter reconnects, which it retries every 10 seconds in the
background. If Postgres goes down, predictions are unaffected and the
inference log has a gap.

---

## 8. Backup and recovery

| Data | Where | Priority |
| --- | --- | --- |
| Model artifacts | Git LFS, `models/artifacts/` | High; the fine-tuned model needs a GPU run to recreate |
| Model registry | `models/registry.json`, in git | High |
| Inference logs, batch jobs, similarity vectors | `postgres_data` volume | Medium |
| Redis | `redis_data` volume | Low; the cache is disposable, queued jobs are not |
| Grafana | `grafana_data` volume | Low; dashboards are provisioned from git |

```bash
docker compose exec -T postgres pg_dump -U mluser mldb | gzip > backup-$(date +%F).sql.gz
gunzip -c backup-2026-09-26.sql.gz | docker compose exec -T postgres psql -U mluser mldb
```

The in-process similarity index (the dev default) is not persisted; it is
empty after a restart. Production uses pgvector, which lives in the Postgres
backup.

Recovery from nothing is a clone, `git lfs pull`, real secrets, `up -d`, and a
restore of the Postgres dump. Every dashboard, alert rule and data source is
provisioned from the repository.
