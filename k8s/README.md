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
| `base/data.yaml` | Redis, Postgres (StatefulSet), the model artifacts PVC |
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

70% rather than 90% because a new pod takes about 20 seconds to load its
models. Scaling at 90% means the replacement capacity arrives after the
overload has already hurt.

Scale-down is deliberately slow, 300 seconds of stabilisation and one pod
every two minutes. Pods are expensive to start here, so flapping costs more
than carrying an idle replica for five minutes. The worker is slower still
because it drains for up to 120 seconds.

CPU is a proxy for what actually matters, which is request latency. To scale
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

## Storage

`model-artifacts` is a `ReadOnlyMany` PVC so the API and worker can share one
copy of the weights. Most cloud block storage only offers `ReadWriteOnce`, and
the claim will not bind there. Two options:

- Bake the artifacts into the image. Simplest, and makes the image the single
  versioned unit, at the cost of a ~400 MB image and a rebuild per model.
- Use a filesystem volume (EFS, Filestore, Azure Files), which supports
  `ReadOnlyMany`.

Postgres uses a `volumeClaimTemplate`, so its storage follows the pod.

## Differences from Compose

**No gateway container.** The Ingress controller already terminates TLS, caps
body size and rate limits at the edge, so the nginx container would be a
second proxy in series for no benefit. The body cap is set to 10 MB to match
the API's own image limit, so oversized uploads are rejected before crossing
the cluster.

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
answer. It hits `/health/live`, which checks nothing external, on purpose: a
liveness probe that checks dependencies restarts healthy pods whenever Redis
hiccups.

`readinessProbe` asks whether traffic should come here, and hits
`/health/ready`, which does check dependencies. A pod that cannot reach its
database should leave the load-balancer pool without being restarted.

## Verifying changes

`tests/unit/test_k8s_manifests.py` renders these and checks the
cross-references: HPAs point at workloads that exist, scaled workloads declare
CPU requests, Service selectors match a pod, volume mounts name a declared
volume, probes use a declared port. They run in the normal suite and need no
cluster.
