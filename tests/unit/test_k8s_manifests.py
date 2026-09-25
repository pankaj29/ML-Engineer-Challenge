"""The Kubernetes manifests.

Schema validation needs a cluster and would not catch what actually breaks
these. A Service whose selector matches no pod is valid YAML, valid against
the schema, and routes to nothing. So does an HPA pointing at a Deployment
that was renamed, or a volumeMount naming a volume that is not declared.

These are the cross-references a human reviewer checks by eye and gets wrong.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "k8s" / "base"

pytestmark = pytest.mark.skipif(not BASE.is_dir(), reason="k8s manifests not present")


def _render() -> list[dict[str, Any]]:
    """Prefer the kustomize output, fall back to reading the files.

    CI may not have kubectl, and the point of these tests is the manifests,
    not the tool.
    """
    try:
        done = subprocess.run(
            ["kubectl", "kustomize", str(BASE)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if done.returncode == 0 and done.stdout.strip():
            return [d for d in yaml.safe_load_all(done.stdout) if d]
    except (OSError, subprocess.SubprocessError):
        pass

    docs: list[dict[str, Any]] = []
    for path in sorted(BASE.glob("*.yaml")):
        if path.name == "kustomization.yaml":
            continue
        docs.extend(d for d in yaml.safe_load_all(path.read_text(encoding="utf-8")) if d)
    return docs


@pytest.fixture(scope="module")
def objects() -> list[dict[str, Any]]:
    docs = _render()
    assert docs, "no manifests were rendered"
    return docs


def _by_kind(objects, kind: str) -> list[dict[str, Any]]:
    return [o for o in objects if o.get("kind") == kind]


def _workloads(objects) -> list[dict[str, Any]]:
    return _by_kind(objects, "Deployment") + _by_kind(objects, "StatefulSet")


def _containers(workload) -> list[dict[str, Any]]:
    return workload["spec"]["template"]["spec"].get("containers", [])


def _name(obj) -> str:
    return obj["metadata"]["name"]


class TestAutoscaling:
    """An HPA that cannot scale is worse than none: it looks like it can."""

    def test_every_hpa_targets_a_workload_that_exists(self, objects) -> None:
        names = {(_name(w), w["kind"]) for w in _workloads(objects)}
        for hpa in _by_kind(objects, "HorizontalPodAutoscaler"):
            ref = hpa["spec"]["scaleTargetRef"]
            assert (
                ref["name"],
                ref["kind"],
            ) in names, (
                f"HPA {_name(hpa)} targets {ref['kind']}/{ref['name']}, which is not deployed"
            )

    def test_scaled_workloads_declare_cpu_requests(self, objects) -> None:
        """Utilisation is a percentage of the request. With no request the
        HPA reports <unknown> and never scales, silently."""
        scaled = {
            h["spec"]["scaleTargetRef"]["name"]
            for h in _by_kind(objects, "HorizontalPodAutoscaler")
        }
        for workload in _workloads(objects):
            if _name(workload) not in scaled:
                continue
            for container in _containers(workload):
                requests = container.get("resources", {}).get("requests", {})
                assert "cpu" in requests, (
                    f"{_name(workload)}/{container['name']} is autoscaled on CPU but "
                    "declares no CPU request"
                )

    def test_a_memory_target_requires_a_memory_request(self, objects) -> None:
        workloads = {_name(w): w for w in _workloads(objects)}
        for hpa in _by_kind(objects, "HorizontalPodAutoscaler"):
            targets = {
                m["resource"]["name"]
                for m in hpa["spec"].get("metrics", [])
                if m.get("type") == "Resource"
            }
            if "memory" not in targets:
                continue
            for container in _containers(workloads[hpa["spec"]["scaleTargetRef"]["name"]]):
                assert "memory" in container.get("resources", {}).get("requests", {})

    def test_min_is_not_above_max(self, objects) -> None:
        for hpa in _by_kind(objects, "HorizontalPodAutoscaler"):
            spec = hpa["spec"]
            assert spec["minReplicas"] <= spec["maxReplicas"]


class TestWiring:
    def test_services_select_a_pod_that_exists(self, objects) -> None:
        pod_labels = [
            w["spec"]["template"]["metadata"].get("labels", {}) for w in _workloads(objects)
        ]
        for service in _by_kind(objects, "Service"):
            selector = service["spec"].get("selector")
            if not selector:
                continue
            matched = any(
                all(labels.get(k) == v for k, v in selector.items()) for labels in pod_labels
            )
            assert matched, f"Service {_name(service)} selects {selector}, which matches no pod"

    def test_volume_mounts_name_a_declared_volume(self, objects) -> None:
        for workload in _workloads(objects):
            pod = workload["spec"]["template"]["spec"]
            declared = {v["name"] for v in pod.get("volumes", [])}
            declared |= {
                c["metadata"]["name"] for c in workload["spec"].get("volumeClaimTemplates", [])
            }
            for container in _containers(workload):
                for mount in container.get("volumeMounts", []):
                    assert mount["name"] in declared, (
                        f"{_name(workload)}/{container['name']} mounts '{mount['name']}', "
                        "which is not declared"
                    )

    def test_claims_referenced_by_pods_are_defined(self, objects) -> None:
        claims = {_name(c) for c in _by_kind(objects, "PersistentVolumeClaim")}
        for workload in _workloads(objects):
            for volume in workload["spec"]["template"]["spec"].get("volumes", []):
                pvc = volume.get("persistentVolumeClaim")
                if pvc:
                    assert (
                        pvc["claimName"] in claims
                    ), f"{_name(workload)} mounts PVC '{pvc['claimName']}', which is not defined"

    def test_secret_keys_referenced_actually_exist(self, objects) -> None:
        secrets = {
            _name(s): set(s.get("stringData", {})) | set(s.get("data", {}))
            for s in _by_kind(objects, "Secret")
        }
        for workload in _workloads(objects):
            for container in _containers(workload):
                for entry in container.get("env", []):
                    ref = entry.get("valueFrom", {}).get("secretKeyRef")
                    if not ref or ref["name"] not in secrets:
                        continue
                    assert (
                        ref["key"] in secrets[ref["name"]]
                    ), f"{_name(workload)} wants {ref['name']}/{ref['key']}, which is not set"

    def test_ingress_backends_point_at_real_services(self, objects) -> None:
        ports = {
            _name(s): {p.get("port") for p in s["spec"].get("ports", [])}
            for s in _by_kind(objects, "Service")
        }
        for ingress in _by_kind(objects, "Ingress"):
            for rule in ingress["spec"].get("rules", []):
                for path in rule.get("http", {}).get("paths", []):
                    backend = path["backend"]["service"]
                    assert backend["name"] in ports, f"ingress routes to unknown {backend['name']}"
                    assert backend["port"]["number"] in ports[backend["name"]]

    def test_probes_use_a_port_the_container_declares(self, objects) -> None:
        for workload in _workloads(objects):
            for container in _containers(workload):
                named = {p.get("name") for p in container.get("ports", [])}
                numbers = {p.get("containerPort") for p in container.get("ports", [])}
                for kind in ("livenessProbe", "readinessProbe", "startupProbe"):
                    http = container.get(kind, {}).get("httpGet")
                    if not http:
                        continue
                    port = http["port"]
                    assert (
                        port in named or port in numbers
                    ), f"{_name(workload)} {kind} uses port {port!r}, which is not declared"


class TestSecurity:
    """The namespace enforces the restricted Pod Security Standard, so a
    container that breaks these will be rejected at admission, not at review."""

    def test_nothing_runs_as_root(self, objects) -> None:
        for workload in _workloads(objects):
            pod = workload["spec"]["template"]["spec"]
            assert (
                pod.get("securityContext", {}).get("runAsNonRoot") is True
            ), f"{_name(workload)} does not require a non-root user"

    def test_privilege_escalation_is_off_and_capabilities_dropped(self, objects) -> None:
        for workload in _workloads(objects):
            for container in _containers(workload):
                ctx = container.get("securityContext", {})
                assert (
                    ctx.get("allowPrivilegeEscalation") is False
                ), f"{_name(workload)}/{container['name']} allows privilege escalation"
                assert ctx.get("capabilities", {}).get("drop") == [
                    "ALL"
                ], f"{_name(workload)}/{container['name']} does not drop all capabilities"

    def test_a_read_only_root_has_somewhere_to_write(self, objects) -> None:
        """readOnlyRootFilesystem with nothing writable mounted fails at
        runtime, usually on the first request rather than at startup.

        Which path it needs depends on the process: the API writes temp files
        to /tmp, Redis writes its append-only file to /data. So this checks
        that a writable mount exists, not that it is called /tmp.
        """
        for workload in _workloads(objects):
            for container in _containers(workload):
                if not container.get("securityContext", {}).get("readOnlyRootFilesystem"):
                    continue
                writable = [
                    m["mountPath"]
                    for m in container.get("volumeMounts", [])
                    if not m.get("readOnly")
                ]
                assert writable, (
                    f"{_name(workload)}/{container['name']} has a read-only root "
                    "filesystem and no writable volume mounted"
                )


class TestAvailability:
    def test_the_api_has_a_disruption_budget(self, objects) -> None:
        """A node drain can otherwise evict every replica at once."""
        budgets = _by_kind(objects, "PodDisruptionBudget")
        assert budgets, "no PodDisruptionBudget: a cluster upgrade can take the API down"

    def test_the_api_surges_rather_than_going_unavailable(self, objects) -> None:
        """Models take about 20 seconds to load, so a pod that is Running is
        not yet a pod that serves. Rolling with maxUnavailable > 0 sheds real
        capacity for that whole window."""
        api = next(d for d in _by_kind(objects, "Deployment") if _name(d) == "ml-api")
        rolling = api["spec"].get("strategy", {}).get("rollingUpdate", {})
        assert rolling.get("maxUnavailable") == 0

    def test_slow_starting_pods_get_a_startup_probe(self, objects) -> None:
        api = next(d for d in _by_kind(objects, "Deployment") if _name(d) == "ml-api")
        container = _containers(api)[0]
        assert "startupProbe" in container, (
            "without a startupProbe the liveness probe restarts the pod while "
            "it is still loading models"
        )
        probe = container["startupProbe"]
        budget = probe["periodSeconds"] * probe["failureThreshold"]
        assert budget >= 60, f"only {budget}s allowed for model loading"


class TestScalingCorrectness:
    def test_the_api_uses_the_shared_similarity_index(self, objects) -> None:
        """This is the setting that makes the API safe to autoscale.

        With the in-memory backend each replica keeps its own index, so an
        image indexed through one pod cannot be found through another. Nothing
        errors; searches just miss.
        """
        config = next(c for c in _by_kind(objects, "ConfigMap") if _name(c) == "mlcv-config")
        assert (
            config["data"].get("SIMILARITY_BACKEND") == "pgvector"
        ), "the API is autoscaled, so the similarity index must be shared"

    def test_postgres_image_supports_pgvector(self, objects) -> None:
        postgres = next(s for s in _by_kind(objects, "StatefulSet") if _name(s) == "postgres")
        image = _containers(postgres)[0]["image"]
        assert "pgvector" in image, (
            f"SIMILARITY_BACKEND=pgvector needs the vector extension, but the " f"image is {image}"
        )

    def test_postgres_is_a_statefulset_not_a_deployment(self, objects) -> None:
        assert not any(_name(d) == "postgres" for d in _by_kind(objects, "Deployment"))
