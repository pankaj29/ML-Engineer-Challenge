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


def _render_overlay(name: str) -> list[dict[str, Any]]:
    """Render one overlay, or skip when kubectl is unavailable.

    Overlays are patches, so unlike the base they cannot be read file by file.
    """
    import subprocess

    path = REPO_ROOT / "k8s" / "overlays" / name
    if not path.is_dir():
        pytest.skip(f"no {name} overlay")
    try:
        done = subprocess.run(
            ["kubectl", "kustomize", str(path)], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        pytest.skip("kubectl is not available")
    if done.returncode != 0:
        pytest.fail(f"the {name} overlay does not render:\n{done.stderr[-600:]}")
    return [d for d in yaml.safe_load_all(done.stdout) if d]


@pytest.fixture(scope="module")
def gpu_objects() -> list[dict[str, Any]]:
    return _render_overlay("gpu")


@pytest.fixture(scope="module")
def canary_objects() -> list[dict[str, Any]]:
    return _render_overlay("canary")


def _by_kind(objects, kind: str) -> list[dict[str, Any]]:
    return [o for o in objects if o.get("kind") == kind]


def _workloads(objects) -> list[dict[str, Any]]:
    return _by_kind(objects, "Deployment") + _by_kind(objects, "StatefulSet")


def _pod_specs(objects) -> list[tuple[str, dict[str, Any]]]:
    """Every pod template, including the ones inside a CronJob."""
    specs = [(_name(w), w["spec"]["template"]["spec"]) for w in _workloads(objects)]
    for cron in _by_kind(objects, "CronJob"):
        specs.append((_name(cron), cron["spec"]["jobTemplate"]["spec"]["template"]["spec"]))
    return specs


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


class TestArtifactDelivery:
    """Weights are not in the image and there is no shared volume, so every
    pod fetches its own copy before the app starts."""

    def test_workloads_that_need_models_fetch_them_first(self, objects) -> None:
        for name in ("ml-api", "worker"):
            workload = next(w for w in _workloads(objects) if _name(w) == name)
            init = workload["spec"]["template"]["spec"].get("initContainers", [])
            assert any(
                c["name"] == "fetch-artifacts" for c in init
            ), f"{name} mounts a models volume but nothing populates it"

    def test_the_models_volume_is_not_a_shared_claim(self, objects) -> None:
        """A ReadOnlyMany PVC is what left the whole stack Pending on a
        cluster whose provisioner only does ReadWriteOnce."""
        claims = {_name(c) for c in _by_kind(objects, "PersistentVolumeClaim")}
        assert "model-artifacts" not in claims

        for name in ("ml-api", "worker"):
            workload = next(w for w in _workloads(objects) if _name(w) == name)
            models = next(
                v for v in workload["spec"]["template"]["spec"]["volumes"] if v["name"] == "models"
            )
            assert "emptyDir" in models, f"{name} still mounts a claim for its weights"

    def test_the_fetcher_writes_where_the_app_reads(self, objects) -> None:
        """A mismatch here gives a pod that downloads 380 MB and then starts
        with an empty artifacts directory."""
        for name in ("ml-api", "worker"):
            workload = next(w for w in _workloads(objects) if _name(w) == name)
            pod = workload["spec"]["template"]["spec"]
            fetcher = next(c for c in pod["initContainers"] if c["name"] == "fetch-artifacts")
            app = pod["containers"][0]

            written = {m["name"] for m in fetcher["volumeMounts"] if not m.get("readOnly")}
            read = {m["name"] for m in app["volumeMounts"]}
            assert "models" in written and "models" in read

            dest = next(e["value"] for e in fetcher["env"] if e["name"] == "ARTIFACT_DEST")
            mount = next(m["mountPath"] for m in fetcher["volumeMounts"] if m["name"] == "models")
            assert dest == mount, f"{name} fetches to {dest} but mounts the volume at {mount}"

    def test_the_source_is_configured(self, objects) -> None:
        config = next(c for c in _by_kind(objects, "ConfigMap") if _name(c) == "mlcv-config")
        assert config["data"].get(
            "ARTIFACT_SOURCE"
        ), "without a source the init container has nowhere to fetch from"


class TestDriftCronJob:
    """The drift check runs here because the inference log is here. A job
    outside the cluster can only read a committed snapshot."""

    def test_the_cronjob_exists_and_does_not_overlap_itself(self, objects) -> None:
        cron = next(iter(_by_kind(objects, "CronJob")), None)
        assert cron is not None, "no scheduled drift check"
        assert (
            cron["spec"]["concurrencyPolicy"] == "Forbid"
        ), "two concurrent runs would both write the history file"

    def test_it_keeps_its_history_across_runs(self, objects) -> None:
        """The cooldown and the confirm-before-acting rule both read this. On
        an emptyDir every run would look like the first."""
        cron = _by_kind(objects, "CronJob")[0]
        pod = cron["spec"]["jobTemplate"]["spec"]["template"]["spec"]
        state = next(v for v in pod["volumes"] if v["name"] == "state")
        assert "persistentVolumeClaim" in state

        claims = {_name(c) for c in _by_kind(objects, "PersistentVolumeClaim")}
        assert state["persistentVolumeClaim"]["claimName"] in claims

    def test_it_reports_rather_than_retrains(self, objects) -> None:
        """--execute would retrain unattended on data nobody inspected, in a
        pod with no GPU and no dataset."""
        cron = _by_kind(objects, "CronJob")[0]
        command = " ".join(
            cron["spec"]["jobTemplate"]["spec"]["template"]["spec"]["containers"][0]["command"]
        )
        assert "--execute" not in command

    def test_it_cannot_run_forever(self, objects) -> None:
        cron = _by_kind(objects, "CronJob")[0]
        assert cron["spec"]["jobTemplate"]["spec"].get("activeDeadlineSeconds")


class TestSecurityAcrossEveryPod:
    """Including init containers and the CronJob, which the earlier checks
    miss because they only look at Deployments."""

    def test_every_container_anywhere_drops_capabilities(self, objects) -> None:
        for name, pod in _pod_specs(objects):
            for container in pod.get("containers", []) + pod.get("initContainers", []):
                ctx = container.get("securityContext", {})
                assert (
                    ctx.get("allowPrivilegeEscalation") is False
                ), f"{name}/{container['name']} allows privilege escalation"
                assert ctx.get("capabilities", {}).get("drop") == [
                    "ALL"
                ], f"{name}/{container['name']} does not drop all capabilities"

    def test_every_pod_anywhere_runs_as_non_root(self, objects) -> None:
        for name, pod in _pod_specs(objects):
            assert pod.get("securityContext", {}).get("runAsNonRoot") is True, name


class TestTheImageHasWhatItNeeds:
    """Caught by deploying, not by reading.

    The image copied models/registry.py but not models/registry.json, so the
    API started, reported its dependencies healthy, and loaded zero models.
    The only symptom was a 503 from /health with every model unhealthy.
    """

    @staticmethod
    def _dockerfile() -> str:
        return (Path(__file__).resolve().parents[2] / "Dockerfile").read_text(encoding="utf-8")

    def test_the_registry_json_is_copied_not_just_the_reader(self) -> None:
        assert (
            "models/registry.json" in self._dockerfile()
        ), "the image has registry.py but not registry.json, so it will load no models"

    def test_the_fetch_script_and_manifest_are_present(self) -> None:
        """The init container runs from this image and verifies against the
        manifest, so both have to be in it."""
        dockerfile = self._dockerfile()
        assert "scripts/fetch_artifacts.py" in dockerfile
        assert "models/artifacts_manifest.json" in dockerfile

    def test_the_mlops_code_the_cronjob_runs_is_present(self) -> None:
        dockerfile = self._dockerfile()
        assert "models/validation/" in dockerfile
        assert "models/pipeline/" in dockerfile


class TestGpuOverlay:
    """Serving on a GPU. Most of the ways this goes wrong are silent: the pod
    runs, answers correctly, and is nowhere near the speed it should be."""

    def test_the_api_requests_a_gpu(self, gpu_objects) -> None:
        api = next(d for d in _by_kind(gpu_objects, "Deployment") if _name(d) == "ml-api")
        container = _containers(api)[0]
        assert container["resources"]["limits"].get("nvidia.com/gpu"), (
            "PREFERRED_RUNTIME is tensorrt but the container requests no GPU, "
            "so it will fall back to CPU and look healthy doing it"
        )

    def test_gpu_requests_and_limits_match(self, gpu_objects) -> None:
        """Kubernetes does not overcommit devices and rejects an unequal pair."""
        api = next(d for d in _by_kind(gpu_objects, "Deployment") if _name(d) == "ml-api")
        pod = api["spec"]["template"]["spec"]
        for container in pod["containers"] + pod.get("initContainers", []):
            resources = container.get("resources", {})
            request = resources.get("requests", {}).get("nvidia.com/gpu")
            limit = resources.get("limits", {}).get("nvidia.com/gpu")
            assert (
                request == limit
            ), f"{container['name']} asks for {request} GPUs and limits {limit}"

    def test_it_is_pinned_to_gpu_nodes(self, gpu_objects) -> None:
        """Without this the scheduler can place it on a CPU node, where
        tensorrt quietly becomes ONNX CPU."""
        api = next(d for d in _by_kind(gpu_objects, "Deployment") if _name(d) == "ml-api")
        pod = api["spec"]["template"]["spec"]
        assert pod.get("nodeSelector"), "no nodeSelector: this can land on a CPU node"
        assert pod.get("tolerations"), "no toleration: GPU pools are usually tainted"

    def test_the_engine_is_built_on_the_serving_node(self, gpu_objects) -> None:
        """A TensorRT engine is compiled for one GPU architecture and one
        TensorRT version, so it cannot be shipped or fetched."""
        api = next(d for d in _by_kind(gpu_objects, "Deployment") if _name(d) == "ml-api")
        init = api["spec"]["template"]["spec"]["initContainers"]
        builder = next((c for c in init if c["name"] == "build-engine"), None)
        assert builder is not None, "nothing builds the engine"
        assert builder["resources"]["limits"].get(
            "nvidia.com/gpu"
        ), "the engine builder has no GPU, so it cannot compile for one"

    def test_the_engine_is_built_after_the_artifacts_arrive(self, gpu_objects) -> None:
        """Init containers run in order. Building before the fetch would
        compile from a file that is not there yet."""
        api = next(d for d in _by_kind(gpu_objects, "Deployment") if _name(d) == "ml-api")
        names = [c["name"] for c in api["spec"]["template"]["spec"]["initContainers"]]
        assert names.index("fetch-artifacts") < names.index("build-engine")

    def test_the_startup_budget_covers_an_engine_build(self, gpu_objects) -> None:
        """Building takes 25 to 90 seconds on top of model loading. Too short
        a budget restarts the pod mid-build, forever."""
        api = next(d for d in _by_kind(gpu_objects, "Deployment") if _name(d) == "ml-api")
        probe = _containers(api)[0]["startupProbe"]
        budget = probe["periodSeconds"] * probe["failureThreshold"]
        assert budget >= 300, f"only {budget}s for fetch, engine build and model load"

    def test_it_does_not_autoscale_on_cpu(self, gpu_objects) -> None:
        """A GPU pod's CPU sits idle while the accelerator saturates, so a CPU
        target never fires and the deployment silently never scales."""
        hpa = next(
            d for d in _by_kind(gpu_objects, "HorizontalPodAutoscaler") if _name(d) == "ml-api"
        )
        kinds = {m.get("type") for m in hpa["spec"]["metrics"]}
        assert "Resource" not in kinds, "still scaling on CPU, which will not move"

    def test_the_runtime_is_actually_tensorrt(self, gpu_objects) -> None:
        config = next(c for c in _by_kind(gpu_objects, "ConfigMap") if _name(c) == "mlcv-config")
        assert config["data"]["PREFERRED_RUNTIME"] == "tensorrt"
        assert config["data"]["DEVICE"] == "cuda"

    def test_every_container_uses_the_gpu_image(self, gpu_objects) -> None:
        """The CPU image has no TensorRT. A pod mixing the two starts and
        fails at the first inference."""
        api = next(d for d in _by_kind(gpu_objects, "Deployment") if _name(d) == "ml-api")
        pod = api["spec"]["template"]["spec"]
        for container in pod["containers"] + pod["initContainers"]:
            assert (
                "gpu" in container["image"]
            ), f"{container['name']} runs {container['image']}, which has no TensorRT"


class TestCanaryOverlay:
    """A slice of real traffic on a new model version. The point is to catch a
    regression that only shows up on the traffic you actually get."""

    def test_stable_and_canary_are_separate_deployments(self, canary_objects) -> None:
        names = {_name(d) for d in _by_kind(canary_objects, "Deployment")}
        assert {"ml-api", "ml-api-canary"} <= names

    def test_they_serve_different_model_versions(self, canary_objects) -> None:
        """Identical sources would make the canary a pointless second copy."""
        stable = next(c for c in _by_kind(canary_objects, "ConfigMap") if _name(c) == "mlcv-config")
        canary = next(
            c for c in _by_kind(canary_objects, "ConfigMap") if _name(c) == "mlcv-canary-config"
        )
        assert canary["data"]["ARTIFACT_SOURCE"] != stable["data"]["ARTIFACT_SOURCE"]

    def test_the_canary_config_wins_for_canary_pods(self, canary_objects) -> None:
        """envFrom applies in order, so the override has to come second."""
        canary = next(
            d for d in _by_kind(canary_objects, "Deployment") if _name(d) == "ml-api-canary"
        )
        sources = [
            e["configMapRef"]["name"]
            for e in _containers(canary)[0]["envFrom"]
            if "configMapRef" in e
        ]
        assert sources.index("mlcv-config") < sources.index("mlcv-canary-config")

    def test_the_canary_takes_a_small_slice(self, canary_objects) -> None:
        ingress = next(d for d in _by_kind(canary_objects, "Ingress") if _name(d) == "mlcv-canary")
        annotations = ingress["metadata"]["annotations"]
        assert annotations["nginx.ingress.kubernetes.io/canary"] == "true"
        weight = int(annotations["nginx.ingress.kubernetes.io/canary-weight"])
        assert 0 < weight <= 25, f"{weight}% is not a canary, it is a rollout"

    def test_the_canary_does_not_autoscale(self, canary_objects) -> None:
        """A canary that scales with traffic stops being a fixed-size sample,
        and its share of the comparison drifts during the experiment."""
        targets = {
            h["spec"]["scaleTargetRef"]["name"]
            for h in _by_kind(canary_objects, "HorizontalPodAutoscaler")
        }
        assert "ml-api-canary" not in targets

    def test_the_canary_can_be_reached_deliberately(self, canary_objects) -> None:
        """Before any real traffic goes near it, someone has to be able to
        send it a request on purpose."""
        ingress = next(d for d in _by_kind(canary_objects, "Ingress") if _name(d) == "mlcv-canary")
        assert ingress["metadata"]["annotations"].get(
            "nginx.ingress.kubernetes.io/canary-by-header"
        )

    def test_the_canary_is_held_to_the_same_security_rules(self, canary_objects) -> None:
        canary = next(
            d for d in _by_kind(canary_objects, "Deployment") if _name(d) == "ml-api-canary"
        )
        pod = canary["spec"]["template"]["spec"]
        assert pod["securityContext"]["runAsNonRoot"] is True
        for container in pod["containers"] + pod["initContainers"]:
            ctx = container["securityContext"]
            assert ctx["allowPrivilegeEscalation"] is False
            assert ctx["capabilities"]["drop"] == ["ALL"]


class TestMigrations:
    """In production the app does not create tables, so something has to.

    Without this the inference log never exists. Writes to it are swallowed by
    design, so nothing errors: drift detection reads an empty table, the
    retraining loop decides there is no drift, and the canary comparison has
    nothing to compare. Every layer reports success.
    """

    def test_something_runs_the_migrations(self, objects) -> None:
        api = next(d for d in _by_kind(objects, "Deployment") if _name(d) == "ml-api")
        init = api["spec"]["template"]["spec"].get("initContainers", [])
        migrate = next((c for c in init if c["name"] == "migrate"), None)
        assert (
            migrate is not None
        ), "nothing applies the schema, and production has create_tables=False"
        assert "alembic" in " ".join(migrate["command"])

    def test_migrations_run_before_anything_else(self, objects) -> None:
        """The API connects to the database at startup. Fetching 380 MB of
        weights first only delays finding out the schema is missing."""
        api = next(d for d in _by_kind(objects, "Deployment") if _name(d) == "ml-api")
        names = [c["name"] for c in api["spec"]["template"]["spec"]["initContainers"]]
        assert names[0] == "migrate", f"migrate is not first: {names}"

    def test_an_initial_migration_is_committed(self) -> None:
        versions = REPO_ROOT / "db" / "migrations" / "versions"
        assert versions.is_dir(), "no migrations directory"
        revisions = [p for p in versions.glob("*.py") if p.name != "__init__.py"]
        assert revisions, "alembic is configured but no migration is committed"

    def test_the_migration_creates_the_inference_log(self) -> None:
        """It is the table the whole feedback loop reads."""
        versions = REPO_ROOT / "db" / "migrations" / "versions"
        text = "".join(p.read_text(encoding="utf-8") for p in versions.glob("*.py"))
        assert "inference_logs" in text

    def test_migrations_leave_the_pgvector_table_alone(self) -> None:
        """similarity_vectors is created by the application, not the ORM, so
        autogenerate sees a table with no model and writes a DROP. That would
        delete the similarity index on the next schema change."""
        env = (REPO_ROOT / "db" / "migrations" / "env.py").read_text(encoding="utf-8")
        assert "similarity_vectors" in env
        assert "include_object" in env

        versions = REPO_ROOT / "db" / "migrations" / "versions"
        for path in versions.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            upgrade = text[text.index("def upgrade") : text.index("def downgrade")]
            assert (
                "similarity_vectors" not in upgrade
            ), f"{path.name} touches the pgvector table in upgrade()"

    def test_the_image_carries_the_alembic_config(self) -> None:
        dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert (
            "alembic.ini" in dockerfile
        ), "the migrate init container runs alembic, which needs its config"
