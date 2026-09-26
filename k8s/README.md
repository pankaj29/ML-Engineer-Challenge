# Kubernetes deployment

The Docker Compose stack is the development environment. This is the same
system for a cluster, with autoscaling and the pieces Compose has no concept
of: disruption budgets, pod security, network policy.

## Deploy

```bash
# 1. Build and push images the cluster can pull
docker build -f docker/Dockerfile.api    -t your-registry/mlcv-api:1.0.0 .
docker build -f docker/Dockerfile.worker -t your-registry/mlcv-worker:1.0.0 .
docker push your-registry/mlcv-api:1.0.0
docker push your-registry/mlcv-worker:1.0.0

# 2. Replace the placeholder secrets. Do not apply config.yaml as it stands.
kubectl create namespace mlcv
kubectl -n mlcv create secret generic mlcv-secrets \
  --from-literal=JWT_SECRET="$(openssl rand -hex 32)" \
  --from-literal=API_KEYS="your-key:pro" \
  --from-literal=POSTGRES_USER=mluser \
  --from-literal=POSTGRES_PASSWORD="$(openssl rand -hex 16)" \
  --from-literal=POSTGRES_DB=mldb \
  --from-literal=DATABASE_URL="postgresql+asyncpg://mluser:PASSWORD@postgres:5432/mldb"

# 3. Apply
kubectl apply -k k8s/base

# 4. Watch it come up
kubectl -n mlcv rollout status deployment/ml-api
kubectl -n mlcv get hpa -w
```

Render without applying: `kubectl kustomize k8s/base`.

## What is here

| File | Contains |
|---|---|
| `base/namespace.yaml` | Namespace, with the restricted Pod Security Standard enforced |
| `base/config.yaml` | ConfigMap and a placeholder Secret |
| `base/data.yaml` | Redis, Postgres (StatefulSet), the model artefacts PVC |
| `base/api.yaml` | API Deployment, Service, PodDisruptionBudget, HPA |
| `base/worker.yaml` | Celery worker Deployment and HPA |
| `base/ingress.yaml` | Ingress and a NetworkPolicy fencing off Postgres |

## Autoscaling

The API scales from 2 to 10 replicas on CPU at 70%, the worker from 1 to 6 at
75%. Both need `metrics-server` in the cluster:

```bash
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl -n mlcv get hpa    # TARGETS shows <unknown> until metrics-server is up
```

The target is 70% because a new pod takes about 20 seconds to load its
models. Scaling at 90% means the replacement capacity arrives after the
overload has already hurt.

Scale-down is slow by design, 300 seconds of stabilisation and one pod every
two minutes. Pods are expensive to start here, so flapping costs more
than carrying an idle replica for five minutes. The worker is slower still
because it drains for up to 120 seconds.

CPU is a proxy for request latency, which is the thing that matters. To scale
on the real thing, install `prometheus-adapter` and use the metrics the API
already publishes at `/api/v1/metrics`:

```yaml
- type: Pods
  pods:
    metric:
      name: http_requests_per_second
    target:
      type: AverageValue
      averageValue: "50"
```

## The similarity index

`SIMILARITY_BACKEND=pgvector` in the ConfigMap is not optional here.

The default in-memory index lives in one process. Under an HPA that means
every replica holds a different index: an image indexed through one pod is
not findable through another, and scaling down discards whatever that pod
held. Nothing errors. Searches just quietly miss.

pgvector puts the vectors in Postgres, which this stack already runs. The
Postgres image is `pgvector/pgvector:pg16` for that reason; plain `postgres`
cannot create the extension.

## Where the weights come from

Not the image, and not a shared volume. An init container downloads them into
an `emptyDir` the app container shares, and verifies every file against
`models/artifacts_manifest.json` before the API is allowed to start.

```
initContainer fetch-artifacts   downloads + verifies -> emptyDir
container api                   reads the emptyDir, read-only
```

`ARTIFACT_SOURCE` in the ConfigMap says where from: `s3://bucket/prefix`,
`https://host/path`, or `file:///path`. Version the prefix and never overwrite
it in place, so rolling a model back is changing that string back.

Three alternatives were considered.

**In the image.** 380 MB, and it ties the model version to the image version:
shipping new weights means redeploying the service, and rolling back a code
change also rolls back the model. Reasonable if you want one versioned unit;
rejected here because model and code move on different schedules.

**A ReadOnlyMany PVC.** Needs a filesystem volume (EFS, Filestore, Azure
Files) that most clusters do not have by default. This is what the manifests
used to do, and applying them to kind is how that was found: the claim never
bound and both workloads sat Pending behind it.

**A ReadWriteOnce PVC.** Binds anywhere, but every pod mounting it must land
on one node, which defeats the autoscaler.

The cost of the current approach is a download per pod start. The fetch script
skips files already present with a matching checksum, so a restart on a warm
volume is a checksum pass, with nothing downloaded.

### Checksums are the point

Serving the wrong weights produces plausible predictions and no error
anywhere. A mismatch fails the init container, so the pod never serves.
Mismatches are not retried: retrying downloads the same wrong bytes.

Regenerate the manifest whenever the weights change:

```bash
python scripts/fetch_artifacts.py \
  --dest models/artifacts \
  --write-manifest models/artifacts_manifest.json
```

A test asserts the committed manifest matches the files on disk, so forgetting
this fails CI, where it is cheap to notice.

## Scheduled drift checks

`drift-watch` is a CronJob, and the reason is data. Drift is computed from the inference log, and the database is in this
namespace. A runner outside the cluster can only read a committed snapshot,
which means deciding on stale data.

It runs weekly, computes drift over the last 7 days against the previous 30,
and asks the retraining pipeline what to do. It reports; it does not retrain.
Training needs a GPU and the dataset, and an unattended retrain on data nobody
inspected is how a working model gets replaced by a worse one.

Its history lives on a small PVC because the pipeline's cooldown and its
confirm-before-acting rule both read it. On an `emptyDir` every run would look
like the first and the cooldown would never apply.

```bash
# Run it now rather than waiting for Monday
kubectl -n mlcv create job --from=cronjob/drift-watch drift-now
kubectl -n mlcv logs job/drift-now
```

## Serving on a GPU

`k8s/overlays/gpu` runs the API through TensorRT on a GPU node. On an A100
the fine-tuned classifier's INT8 engine runs at 0.92 ms and 1068 img/s, against
19.8 img/s through ONNX Runtime (fp32) on the development laptop's CPU
(`benchmarks/reports/tensorrt.json`, `BENCHMARKS.md`). The engine has to be
built on the serving node: it is tied to one GPU architecture and TensorRT
version.

```bash
docker build -f docker/Dockerfile.gpu -t your-registry/mlcv-api-gpu:1.0.0 .
kubectl apply -k k8s/overlays/gpu
```

Needs a GPU node, the NVIDIA device plugin or GPU Operator, and for the HPA,
dcgm-exporter behind prometheus-adapter.

Three things in it are not obvious.

**The engine is built at pod start, on the node that will serve it.** A
TensorRT engine is compiled for one GPU architecture and one TensorRT version:
one built on an A100 will not load on an L4, and one built with 11.3 will not
load under 11.4. So it cannot go in the image or come from a bucket. A
`build-engine` init container runs after the artefact fetch and compiles the
INT8 QDQ graph. Budget about 25 to 90 seconds, which is why the startup probe
allows 600.

**It scales on GPU utilisation, not CPU.** A GPU pod's CPU sits near idle
while the accelerator saturates, so the base's CPU target would never fire and
the deployment would never scale. The metric is
`DCGM_FI_DEV_GPU_UTIL`. Without dcgm-exporter it is unavailable and the HPA
holds at `minReplicas`, which is the safe failure.

**The node selector is load-bearing.** Without it the scheduler will place a
pod on a CPU node, `PREFERRED_RUNTIME=tensorrt` falls back to ONNX CPU, and
the pod is healthy and correct and fifteen times slower than the dashboard
suggests.

This overlay has not been run on hardware. It renders and is covered by
structural tests, but no GPU was available.

## Canary releases

`k8s/overlays/canary` runs a second deployment on a different model version
taking 5% of traffic.

```bash
kubectl apply -k k8s/overlays/canary

# Hit the canary directly, before any real traffic reaches it
curl -H "X-Canary: always" http://your-host/api/v1/health
```

This is the control the rest of the system lacks. Validation and the
regression gate both run before a model is live, against held-out data.
Neither can tell you how it behaves on your real traffic, which is where a
model usually disappoints.

Both deployments write to the same inference log with their model version
recorded, so after a canary period the existing A/B machinery compares them on
real traffic:

```bash
python -m models.validation.ab_test --champion 1.0.0 --challenger 1.1.0
```

Promote by setting `ARTIFACT_SOURCE` in `mlcv-config` to the canary's value,
rolling `ml-api`, then deleting the overlay. Roll back by deleting the
overlay: stable was never touched.

The canary has no HPA. One that scales with traffic stops being a fixed-size
sample, and its share of the comparison drifts mid-experiment.

## Differences from Compose

**No gateway container.** The Ingress controller already terminates TLS, caps
body size and rate limits at the edge, so the nginx container would be a
second proxy in series for no benefit. The body cap is 10 MB, matching the
API's own image limit, so oversized uploads are rejected before crossing the
cluster.

That differs from the Compose gateway by design, and it changes what a caller
sees. `docker/nginx/nginx.conf` allows 12 MB so an 11 MB upload reaches the API
and gets its JSON `IMAGE_TOO_LARGE` error; here the same upload gets the
ingress controller's plain HTML 413. Compose favours the clearer message, the
cluster favours not carrying rejected bytes across the network.

**Redis is not persisted.** It holds the cache and the rate-limit buckets, and
both rebuild in seconds. A single-replica Redis on a PVC is a slower restart,
not a more available one. If you move session state or job results into it,
revisit that.

**Postgres is a single StatefulSet.** Fine for a demo, not for production.
Use a managed database or an operator such as CloudNativePG, which handles
failover, backups and point-in-time recovery. Those are not things to
hand-roll.

## Probes

Three probes on the API, answering three different questions.

`startupProbe` asks whether the models have loaded, and suspends the other two
until they have. Budget is 150 seconds (30 failures x 5s). Without it the
liveness probe restarts the pod mid-load, forever.

`livenessProbe` asks whether the process is wedged, and restart is the only
answer. It hits `/health/live`, which checks nothing external: a liveness
probe that checks dependencies restarts healthy pods whenever Redis
hiccups.

`readinessProbe` asks whether traffic should come here, and hits
`/health/ready`, which does check dependencies. A pod that cannot reach its
database should leave the load-balancer pool without being restarted.

## Verified on a cluster

These are not manifests that only render. Applied to a kind cluster
(Kubernetes v1.33.1) with the `kind` overlay, the stack came up and served a
real request.

```
NAME                              READY   STATUS
artifact-server-56db9c7c6-xglrp   1/1     Running
ml-api-7d9f4c8b96-2qkxp           1/1     Running
postgres-0                        1/1     Running
redis-6cdbfb94bf-th2cg            1/1     Running
worker-56d7b9c7fd-c6vhk           1/1     Running

status: healthy
  cache                      healthy
  database                   healthy
  model:classification       healthy
  model:detection            healthy
  model:similarity           healthy
```

End to end, through the cluster: a token from `POST /api/v1/auth/token`, used
to authenticate an upload to `/api/v1/classify/upload`, which returned
Labrador retriever at 0.397 for `samples/dog.jpg`.

What else the run confirmed:

- **The init container fetches and verifies.** All 11 artefacts pulled over
  HTTP from the in-cluster store, every checksum checked, 382.5 MB. The retry
  path fired for real: the first request hit connection-refused before the
  server was ready, and the retry succeeded.
- **The HPA scales.** With metrics-server present the targets resolved, and
  dropping the memory target to 1% took ml-api from one pod to two inside a
  minute. Restored afterwards; the manifest was untouched.
- **Rollouts do not shed capacity.** `maxUnavailable: 0` held: the old pod
  terminated only after the new one was Ready.
- **pgvector provisions itself.** The API created `similarity_vectors` with a
  `vector(2048)` column against in-cluster Postgres, extension 0.8.6.
- **The drift CronJob runs where the data is.** `kubectl create job
  --from=cronjob/drift-watch` completed: it queried the inference log, wrote a
  report, and the pipeline decided to skip. That is the whole reason it is a
  CronJob and not a GitHub Action.

### Three bugs this found

None of these were visible from reading the manifests.

**`models/registry.json` was not in the image.** The Dockerfile copied
`registry.py` but not the registry it reads. The API started, reported cache
and database healthy, and loaded zero models. The only symptom was a 503 from
`/health` with every model unhealthy. A test now asserts the Dockerfile copies
it.

**`CREATE EXTENSION IF NOT EXISTS` is not atomic.** Two replicas starting
together both found the vector extension missing, both tried to create it, and
the loser took a unique violation on `pg_extension_name_index` and came up with
similarity degraded. Schema setup now tolerates losing that race inside a
savepoint, and still propagates anything else, such as a permissions error.

**`hostPath` is forbidden by the restricted Pod Security Standard.** An earlier
version of this overlay mounted the repo's artefacts directly and every
ReplicaSet was rejected at admission. That is the control working. Serving the
files over HTTP from an in-cluster pod keeps the security posture identical to
production and exercises the real network fetch path, not a `file://`
shortcut.

An earlier run also confirmed why the `ReadOnlyMany` PVC had to go: on
local-path storage the claim never bound and both workloads sat Pending behind
it.

### What this did not verify

A single node cannot exercise the multi-node behaviour the base targets, and
TensorRT is not involved: these are the CPU ONNX runtimes. The artefact store
here is a pod serving static files, not S3, so the `s3://` branch of the fetch
script is covered by unit tests, not by this run.

## Verifying changes

`tests/unit/test_k8s_manifests.py` renders these and checks the
cross-references: HPAs point at workloads that exist, scaled workloads declare
CPU requests, Service selectors match a pod, volume mounts name a declared
volume, probes use a declared port. They run in the normal suite and need no
cluster.
