"""Shared pytest fixtures.

Plain English:
    A fixture is a reusable piece of test setup. Everything here exists so
    that a test can say "give me a working API client" in one line, without
    needing Redis, PostgreSQL or a GPU to be running.

The guiding rule: **unit and integration tests must run on a laptop with
nothing installed but Python.** A test suite that needs a database to be up is
a test suite that gets skipped, and a skipped test protects nothing.

How each external dependency is replaced:

* **Redis** -> ``fakeredis``, an in-memory implementation of the real protocol.
* **PostgreSQL** -> SQLite via ``aiosqlite``, created fresh per test and thrown
  away afterwards.
* **Models** -> a tiny fake runtime returning fixed arrays, so route and
  service tests run in milliseconds instead of loading 100 MB of ONNX.

Tests that genuinely need a real model are marked ``@pytest.mark.slow`` and
skip automatically when the artifacts are missing.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Environment must be set BEFORE any api.* import, because api.config builds
# its settings singleton at import time.
#
# The identity settings below are ASSIGNED, not setdefault-ed. Tests hard-code
# "test-pro-key" and friends, so if an ambient API_KEYS exists the fixtures
# authenticate against a key set the app has never heard of and every
# authenticated request returns 401 - which reads as 50+ unrelated assertion
# failures rather than "your key is wrong".
#
# That is not hypothetical: CI exported `API_KEYS=ci-key:pro`, setdefault
# silently kept it, and the pipeline failed from its first run.
os.environ["ENVIRONMENT"] = "test"
os.environ["AUTH_ENABLED"] = "true"
os.environ["API_KEYS"] = "test-free-key:free,test-pro-key:pro,test-ent-key:enterprise"
os.environ["JWT_SECRET"] = "test-secret-key-that-is-long-enough-for-validation-32"

# The rest stay setdefault: they are tunables a developer may legitimately
# override for one run, and nothing asserts on their exact value.
os.environ.setdefault("CACHE_ENABLED", "false")
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("EAGER_MODEL_LOAD", "false")
os.environ.setdefault("LOG_LEVEL", "CRITICAL")
os.environ.setdefault("METRICS_ENABLED", "true")

from PIL import Image

from api.config import Settings, get_settings


# ---------------------------------------------------------------------------
# Event loop
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """Session-scoped event loop, so async fixtures can be session-scoped too."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
@pytest.fixture
def settings() -> Settings:
    """Fresh settings for the test environment."""
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
def strict_settings() -> Settings:
    """Settings with tight limits, for testing validation boundaries."""
    return Settings(
        environment="test",
        max_image_bytes=50_000,
        max_image_pixels=100_000,
        min_image_dimension=32,
        max_image_dimension=1024,
        allowed_image_formats="JPEG,PNG",
        max_batch_size=4,
    )


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
def make_image(
    width: int = 224,
    height: int = 224,
    color: tuple[int, int, int] = (128, 64, 32),
    fmt: str = "PNG",
    mode: str = "RGB",
) -> bytes:
    """Build an in-memory test image.

    Generating images rather than committing fixture files keeps the repo
    small and lets a test ask for the exact size or format it needs.
    """
    img = Image.new(mode, (width, height), color if mode == "RGB" else color[0])
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    return buffer.getvalue()


def make_noise_image(width: int = 224, height: int = 224, fmt: str = "PNG") -> bytes:
    """Build a random-noise image.

    Flat-colour images compress to almost nothing, which makes them useless
    for testing size limits. Noise does not compress.
    """
    array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
def sample_image() -> bytes:
    """A valid 224x224 PNG."""
    return make_image()


@pytest.fixture
def sample_image_b64(sample_image: bytes) -> str:
    """The sample image, base64-encoded for a JSON body."""
    return base64.b64encode(sample_image).decode()


@pytest.fixture
def sample_jpeg() -> bytes:
    """A valid JPEG, for format-handling tests."""
    return make_image(fmt="JPEG")


@pytest.fixture
def tiny_image() -> bytes:
    """An image below the minimum dimension."""
    return make_image(width=8, height=8)


@pytest.fixture
def grayscale_image() -> bytes:
    """A single-channel image, which must be converted to RGB."""
    return make_image(mode="L")


@pytest.fixture
def corrupt_image() -> bytes:
    """Bytes that begin like a PNG but are truncated."""
    return make_image()[:60]


@pytest.fixture
def not_an_image() -> bytes:
    """Bytes that are definitely not an image."""
    return b"This is plain text, not an image file at all. " * 5


# ---------------------------------------------------------------------------
# Fake model runtime
# ---------------------------------------------------------------------------
# Highest class score a fake detector will emit. Must stay below any threshold
# a test uses to assert "no detections survive". See FakeRuntime.infer.
MAX_FAKE_SCORE = 0.99


class FakeRuntime:
    """A stand-in for a real model.

    Returns a deterministic array shaped like the real thing, so tests can
    assert on postprocessing logic without loading 100 MB of weights. The
    output is derived from the input sum, so different inputs give different
    (but repeatable) answers — which is what makes cache tests meaningful.
    """

    def __init__(
        self,
        output_shape: tuple[int, ...] = (1, 1000),
        runtime_format: Any = None,
        device: str = "cpu",
        fail: bool = False,
        delay: float = 0.0,
    ) -> None:
        from api.models.schemas import RuntimeFormat

        self.output_shape = output_shape
        self.format = runtime_format or RuntimeFormat.ONNX
        self.device = device
        self.fail = fail
        self.delay = delay
        self.call_count = 0
        self.last_input: np.ndarray | None = None

    def infer(self, inputs: np.ndarray) -> list[np.ndarray]:
        self.call_count += 1
        self.last_input = inputs
        if self.fail:
            raise RuntimeError("simulated inference failure")
        if self.delay:
            import time

            time.sleep(self.delay)

        batch = inputs.shape[0]
        shape = (batch, *self.output_shape[1:])
        # Deterministic but input-dependent, via a seeded generator.
        seed = int(abs(float(np.sum(inputs))) * 1000) % (2**31)
        rng = np.random.default_rng(seed)
        output = rng.standard_normal(shape).astype(np.float32)

        # A detector-shaped output (batch, 4 + classes, anchors) must look like
        # the real thing: rows 0-3 are box geometry in pixels, and the class
        # scores that follow are probabilities. Returning raw normal values
        # there would not exercise the real postprocessing path.
        #
        # Scores are capped at MAX_FAKE_SCORE rather than 1.0. A detector
        # output is 80 classes x 8400 anchors = 672,000 samples, so with a
        # uniform [0, 1) draw the chance that at least one exceeds a
        # "nothing should survive this" threshold of 0.999999 is about 49%
        # (expected count 0.67). That made test_high_threshold_returns_nothing
        # a coin flip that depended on the seed, which is derived from the
        # input sum and so differs between machines. Capping the draw makes
        # the assertion deterministic without weakening it: every other
        # detection test uses thresholds well below this.
        if len(shape) == 3 and shape[1] > 4:
            output[:, :4, :] = rng.uniform(10.0, 600.0, (batch, 4, shape[2]))
            output[:, 4:, :] = rng.uniform(0.0, MAX_FAKE_SCORE, (batch, shape[1] - 4, shape[2]))
        return [output]

    def close(self) -> None:
        pass


@pytest.fixture
def fake_classifier_runtime() -> FakeRuntime:
    """Fake runtime shaped like a 1000-class classifier."""
    return FakeRuntime(output_shape=(1, 1000))


@pytest.fixture
def fake_detector_runtime() -> FakeRuntime:
    """Fake runtime shaped like YOLOv8: (batch, 4 + 80 classes, anchors)."""
    return FakeRuntime(output_shape=(1, 84, 8400))


@pytest.fixture
def fake_embedder_runtime() -> FakeRuntime:
    """Fake runtime shaped like a 2048-d embedding model."""
    return FakeRuntime(output_shape=(1, 2048))


def make_model_entry(
    name: str = "test-model",
    version: str = "1.0.0",
    task: str = "classification",
    **overrides: Any,
) -> Any:
    """Build a :class:`~api.services.model_service.ModelEntry` for tests."""
    from api.models.schemas import TaskType
    from api.services.model_service import ModelEntry

    defaults: dict[str, Any] = {
        "name": name,
        "version": version,
        "task": TaskType(task),
        "artifacts": {"onnx": f"{name}.onnx"},
        "preprocess": "imagenet_224",
        "num_classes": 1000,
        "input_shape": [1, 3, 224, 224],
        "is_default": True,
    }
    defaults.update(overrides)
    return ModelEntry(**defaults)


@pytest.fixture
def fake_model(fake_classifier_runtime: FakeRuntime) -> Any:
    """A LoadedModel wrapping the fake classifier runtime."""
    import time

    from api.services.model_service import LoadedModel

    return LoadedModel(
        entry=make_model_entry(),
        runtime=fake_classifier_runtime,
        labels=[f"label_{i}" for i in range(1000)],
        loaded_at=time.time(),
    )


class FakeModelService:
    """A ModelService that serves preloaded fakes instead of reading disk."""

    def __init__(self, models: dict[str, Any] | None = None) -> None:
        self.models = models or {}
        self.device = "cpu"
        self.resolve_calls: list[tuple] = []

    def artifact_fingerprint(self, entry: Any) -> str:
        """Stand-in for the real content hash.

        Constant, because these fakes have no artifact on disk. Tests that care
        about invalidation set this explicitly.
        """
        return getattr(self, "fingerprint", "fake")

    def resolve(self, task: Any, name: str | None = None, version: str | None = None) -> Any:
        from api.exceptions import ModelNotFoundError

        self.resolve_calls.append((task, name, version))
        key = task.value if hasattr(task, "value") else str(task)
        model = self.models.get(key)
        if model is None:
            raise ModelNotFoundError(f"no model registered for {key}")
        if name and name != model.entry.name:
            raise ModelNotFoundError(f"no model named {name}")
        if version and version != "latest" and version != model.entry.version:
            raise ModelNotFoundError(f"no version {version}")
        return model.entry

    def load(self, entry: Any, requested: Any = None) -> Any:
        key = entry.task.value
        return self.models[key]

    async def load_async(self, entry: Any, requested: Any = None) -> Any:
        return self.load(entry, requested)

    async def get(
        self, task: Any, name: str | None = None, version: str | None = None, runtime: Any = None
    ) -> Any:
        return self.load(self.resolve(task, name, version), runtime)

    def list_entries(self) -> list[Any]:
        return [m.entry for m in self.models.values()]

    def default_keys(self) -> dict[str, str]:
        return {k: m.entry.key for k, m in self.models.items()}

    def health(self) -> dict[str, Any]:
        return {
            "registered": len(self.models),
            "loaded": len(self.models),
            "loaded_keys": [f"{m.entry.key}:auto" for m in self.models.values()],
            "failures": {},
            "device": "cpu",
        }

    def unload_all(self) -> None:
        self.models.clear()

    async def warmup(self) -> dict[str, str]:
        return dict.fromkeys(self.models, "loaded")

    def reload_registry(self) -> None:
        pass

    def _artifact_path(self, entry: Any, fmt: str) -> Path | None:
        return None


@pytest.fixture
def fake_model_service(
    fake_classifier_runtime: FakeRuntime,
    fake_detector_runtime: FakeRuntime,
    fake_embedder_runtime: FakeRuntime,
) -> FakeModelService:
    """A model service with one fake model per task."""
    import time

    from api.services.model_service import LoadedModel

    return FakeModelService(
        {
            "classification": LoadedModel(
                entry=make_model_entry("test-classifier", task="classification"),
                runtime=fake_classifier_runtime,
                labels=[f"class_{i}" for i in range(1000)],
                loaded_at=time.time(),
            ),
            "detection": LoadedModel(
                entry=make_model_entry(
                    "test-detector",
                    task="detection",
                    preprocess="yolo_640",
                    num_classes=80,
                    input_shape=[1, 3, 640, 640],
                ),
                runtime=fake_detector_runtime,
                labels=[f"object_{i}" for i in range(80)],
                loaded_at=time.time(),
            ),
            "similarity": LoadedModel(
                entry=make_model_entry("test-embedder", task="similarity", num_classes=None),
                runtime=fake_embedder_runtime,
                labels=[],
                loaded_at=time.time(),
            ),
        }
    )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
@pytest.fixture
async def fake_cache() -> Any:
    """A CacheService backed by in-memory fakeredis.

    Exercises the real code path — serialisation, TTLs, key building — without
    needing a Redis server.
    """
    from api.services.cache_service import CacheService

    service = CacheService()
    try:
        import fakeredis.aioredis

        service._client = fakeredis.aioredis.FakeRedis(decode_responses=True)
        service._available = True
        service.settings = Settings(environment="test", cache_enabled=True)
    except ImportError:
        pytest.skip("fakeredis is not installed")

    yield service
    await service.close()


class NullCache:
    """A cache that always misses. Used to test the cache-disabled path."""

    def __init__(self) -> None:
        from api.services.cache_service import CacheStats

        self.stats = CacheStats()
        self.available = False
        self.set_calls: list[tuple] = []

    async def get(self, key: str) -> None:
        return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        self.set_calls.append((key, value))
        return False

    async def delete(self, key: str) -> bool:
        return False

    async def invalidate_model(self, model_key: str) -> int:
        return 0

    async def connect(self) -> bool:
        return False

    async def close(self) -> None:
        pass

    async def health(self) -> dict[str, Any]:
        return {"status": "disabled"}


@pytest.fixture
def null_cache() -> NullCache:
    """A cache that never stores anything."""
    return NullCache()


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
@pytest.fixture
async def test_db() -> Any:
    """A DatabaseService backed by an in-memory SQLite database.

    Each test gets a brand-new database, created and destroyed around it, so
    tests cannot pollute one another through leftover rows.
    """
    from api.services.db_service import DatabaseService

    service = DatabaseService(
        Settings(
            environment="test",
            database_url="sqlite+aiosqlite:///:memory:",
            database_pool_size=1,
            database_max_overflow=0,
        )
    )
    try:
        connected = await service.connect(create_tables=True)
    except Exception:
        pytest.skip("aiosqlite is not installed")
    if not connected:
        pytest.skip("could not create the in-memory test database")

    yield service
    await service.close()


# ---------------------------------------------------------------------------
# Inference service and API client
# ---------------------------------------------------------------------------
@pytest.fixture
def inference_service(fake_model_service: FakeModelService, null_cache: NullCache) -> Any:
    """An InferenceService wired to fakes."""
    from api.services.inference_service import InferenceService

    return InferenceService(fake_model_service, null_cache)  # type: ignore[arg-type]


@pytest.fixture
def api_client(fake_model_service: FakeModelService, null_cache: NullCache) -> Iterator[Any]:
    """A FastAPI TestClient with every dependency replaced by a fake.

    ``dependency_overrides`` is FastAPI's built-in mechanism for exactly this:
    the app is wired normally, and only the leaves are swapped.
    """
    from fastapi.testclient import TestClient

    from api.main import create_app
    from api.services.cache_service import get_cache_service
    from api.services.inference_service import InferenceService, get_inference_service
    from api.services.model_service import get_model_service

    app = create_app(testing=True)
    service = InferenceService(fake_model_service, null_cache)  # type: ignore[arg-type]

    app.dependency_overrides[get_model_service] = lambda: fake_model_service
    app.dependency_overrides[get_cache_service] = lambda: null_cache
    app.dependency_overrides[get_inference_service] = lambda: service

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client

    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Headers for an authenticated pro-tier request."""
    return {"X-API-Key": "test-pro-key"}


@pytest.fixture
def free_tier_headers() -> dict[str, str]:
    """Headers for an authenticated free-tier request."""
    return {"X-API-Key": "test-free-key"}


# ---------------------------------------------------------------------------
# Real-artifact gating
# ---------------------------------------------------------------------------
# First bytes of a Git LFS pointer file, per the LFS spec.
_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def is_usable_artifact(path: Path) -> bool:
    """True if `path` is a real model file rather than an LFS pointer.

    Model artifacts are stored in Git LFS. A clone made without git-lfs - or a
    CI checkout missing `lfs: true` - leaves a ~130-byte TEXT file at each
    artifact path containing an oid, not the model.

    Checking only `.exists()` passes those, and the failure then surfaces much
    later as `ModelLoadError: could not be loaded in any available format`,
    repeated across every test that touches a model. Detecting the pointer
    here turns 24 cryptic errors into a skip that names the cause.
    """
    if not path.is_file():
        return False
    try:
        with path.open("rb") as handle:
            head = handle.read(len(_LFS_POINTER_PREFIX))
    except OSError:
        return False
    return head != _LFS_POINTER_PREFIX


@pytest.fixture(scope="session")
def real_models_available() -> bool:
    """True when real model artifacts are present AND are not LFS pointers."""
    registry = REPO_ROOT / "models" / "registry.json"
    if not registry.exists():
        return False
    try:
        data = json.loads(registry.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    models = data.get("models", [])
    if not models:
        return False
    artifacts_dir = REPO_ROOT / "models" / "artifacts"
    return any(
        is_usable_artifact(artifacts_dir / rel)
        for entry in models
        for rel in entry.get("artifacts", {}).values()
    )


@pytest.fixture
def require_real_models(real_models_available: bool) -> None:
    """Skip a test unless real artifacts are present and usable.

    With REQUIRE_MODELS set (the CI step that checks quantized fidelity), a
    missing artifact fails instead: a skipped test there reports green while
    checking nothing, which is how a broken INT8 detector went unnoticed.
    """
    if real_models_available:
        return
    if os.getenv("REQUIRE_MODELS"):
        pytest.fail("REQUIRE_MODELS is set but real model artifacts are missing or LFS pointers")
    # Name the likely cause. The two reasons differ, and the wrong advice
    # sends someone re-exporting models when all they needed was a pull.
    artifacts = REPO_ROOT / "models" / "artifacts"
    pointers = [
        f.name
        for f in artifacts.glob("*")
        if f.suffix in {".onnx", ".pt", ".engine"} and not is_usable_artifact(f)
    ]
    if pointers:
        pytest.skip(
            f"model artifacts are Git LFS pointers, not real files "
            f"({len(pointers)} of them). Fetch with: git lfs pull"
        )
    pytest.skip("real model artifacts are not prepared; run: python scripts/prepare_models.py")


def pytest_configure(config: Any) -> None:
    """Register custom markers so ``-m`` filtering works and -W error is clean."""
    config.addinivalue_line("markers", "slow: takes more than a second")
    config.addinivalue_line("markers", "integration: crosses a component boundary")
    config.addinivalue_line("markers", "performance: measures speed or memory")
    config.addinivalue_line("markers", "requires_models: needs real model artifacts")
