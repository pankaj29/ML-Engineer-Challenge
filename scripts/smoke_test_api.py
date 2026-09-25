"""Exercise every documented endpoint against a running deployment.

Plain English:
    The manual pass a reviewer would do by hand, written down so it is
    repeatable. Start the stack, run this, read the table.

By default it goes through the gateway on port 80, so nginx, routing, body
limits and the whole middleware stack are in the path. That is the deployment
a caller actually talks to, which is why it is the default.

`--target direct` skips nginx and hits the API's own port 8000 instead. Use it
when you only want to know whether the application works, or when the gateway
is not running. If the two disagree, the difference is nginx: its body limit,
its timeouts or its proxy headers. `--target both` runs the suite twice and
reports each separately, which is the quickest way to find that out.

Refusing correctly counts as passing. A 401 where 200 was expected is a
failure; a 401 where 401 was expected is the system working, so the bad-key
and SSRF checks assert on the rejection rather than treating any non-200 as
broken.

It found a real gap the first time it ran: `POST /auth/token` returned 401
because docker-compose left JWT_SECRET empty, so a documented endpoint could
not work on the default stack.

Usage::

    docker compose up -d
    python scripts/smoke_test_api.py

    # the API alone, no gateway in front
    python scripts/smoke_test_api.py --target direct

    # both, to see whether nginx changes any answer
    python scripts/smoke_test_api.py --target both

    # somewhere else entirely
    python scripts/smoke_test_api.py --base-url https://api.example.com --api-key KEY

Exit code is 0 when every check passes, 1 otherwise, so it can gate a deploy.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent

GATEWAY_URL = "http://localhost"
DIRECT_URL = "http://localhost:8000"

# (target, group, check, ok, detail)
results: list[tuple[str, str, str, bool, str]] = []

BASE = ""
KEY = ""
TARGET = ""


def check(name: str, ok: bool, detail: str = "", group: str = "") -> None:
    results.append((TARGET, group, name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f"  {detail}" if detail else ""))


def run_suite(target: str, base_url: str, key: str, image_path: Path, timeout: float) -> None:
    """Run every check against one deployment.

    The suite is identical for the gateway and for the API's own port, which is
    the point: any difference between the two runs is nginx, not the test.
    """
    global BASE, KEY, TARGET
    TARGET = target
    BASE = f"{base_url.rstrip('/')}/api/v1"
    KEY = key

    print(f"\n{'#' * 60}")
    print(f"# {target}: {BASE}")
    print("#" * 60)

    image = image_path.read_bytes()
    b64 = base64.b64encode(image).decode()

    client = httpx.Client(timeout=timeout)
    hk = {"X-API-Key": KEY}

    # ---------------------------------------------------------------- health
    print("\nHealth and metrics (public)")
    for path in ("/health/live", "/health/ready", "/health"):
        r = client.get(f"{BASE}{path}")
        check(f"GET {path}", r.status_code in (200, 503), f"{r.status_code}", "health")
        if path == "/health":
            body = r.json()
            comps = {c["name"]: c["status"] for c in body.get("components", [])}
            print(f"         status={body.get('status')}  {comps}")

    r = client.get(f"{BASE}/metrics")
    check(
        "GET /metrics",
        r.status_code == 200 and "http_requests_total" in r.text,
        f"{r.status_code}, {len(r.text)} bytes",
        "health",
    )

    # ------------------------------------------------------------------ auth
    print("\nAuthentication")
    r = client.post(f"{BASE}/auth/token", json={"api_key": KEY})
    token_ok = r.status_code == 200 and "access_token" in r.json()
    check("POST /auth/token with a valid key", token_ok, f"{r.status_code}", "auth")
    token = r.json().get("access_token", "") if token_ok else ""
    if token_ok:
        b = r.json()
        print(f"         tier={b['tier']} expires_in={b['expires_in']}s")

    r = client.post(f"{BASE}/auth/token", json={"api_key": "not-a-real-key"})
    check(
        "POST /auth/token with a bad key is refused",
        r.status_code == 401,
        f"{r.status_code}",
        "auth",
    )

    r = client.get(f"{BASE}/models")
    check("an unauthenticated call is refused", r.status_code == 401, f"{r.status_code}", "auth")

    if token:
        r = client.get(f"{BASE}/models", headers={"Authorization": f"Bearer {token}"})
        check("the issued token authenticates", r.status_code == 200, f"{r.status_code}", "auth")

    # ---------------------------------------------------------- classification
    print("\nClassification")
    r = client.post(f"{BASE}/classify", headers=hk, json={"image_base64": b64, "top_k": 3})
    ok = r.status_code == 200
    check("POST /classify (base64)", ok, f"{r.status_code}", "classify")
    if ok:
        body = r.json()
        top = body["predictions"][0]
        print(
            f"         {body['model']['name']}:{body['model']['version']} "
            f"[{body['model']['runtime']}] -> {top['label']} {top['confidence']:.3f} "
            f"({body['timing']['total_ms']} ms)"
        )
        check(
            "top_k is honoured",
            len(body["predictions"]) == 3,
            f"{len(body['predictions'])} predictions",
            "classify",
        )

    r = client.post(
        f"{BASE}/classify/upload",
        headers=hk,
        files={"file": (image_path.name, image, "image/jpeg")},
        data={"top_k": "2"},
    )
    check("POST /classify/upload (multipart)", r.status_code == 200, f"{r.status_code}", "classify")

    # ---------------------------------------------------------------- detection
    print("\nDetection")
    r = client.post(f"{BASE}/detect", headers=hk, json={"image_base64": b64})
    ok = r.status_code == 200
    check("POST /detect (base64)", ok, f"{r.status_code}", "detect")
    if ok:
        body = r.json()
        print(f"         {len(body['detections'])} detections, " f"{body['timing']['total_ms']} ms")
        if body["detections"]:
            d = body["detections"][0]
            check(
                "a detection carries a box",
                all(k in d["box"] for k in ("x1", "y1", "x2", "y2")),
                str(d.get("label")),
                "detect",
            )

    r = client.post(
        f"{BASE}/detect/upload", headers=hk, files={"file": (image_path.name, image, "image/jpeg")}
    )
    check("POST /detect/upload (multipart)", r.status_code == 200, f"{r.status_code}", "detect")

    # --------------------------------------------------------------- similarity
    print("\nSimilarity")
    r = client.post(f"{BASE}/similarity/embed", headers=hk, json={"image_base64": b64})
    ok = r.status_code == 200
    check("POST /similarity/embed", ok, f"{r.status_code}", "similarity")
    if ok:
        print(f"         {len(r.json()['embedding'])}-dimensional embedding")

    r = client.post(
        f"{BASE}/similarity/index",
        headers=hk,
        json={"image_base64": b64, "label": "smoke-test", "image_id": "smoke-1"},
    )
    indexed = r.status_code == 201
    check("POST /similarity/index", indexed, f"{r.status_code}", "similarity")

    r = client.post(f"{BASE}/similarity/search", headers=hk, json={"image_base64": b64, "top_k": 3})
    ok = r.status_code == 200
    check("POST /similarity/search", ok, f"{r.status_code}", "similarity")
    if ok:
        body = r.json()
        print(f"         {body['count']} hits, index size {body['index_size']}")
        if indexed and body["count"]:
            check(
                "the indexed image is found again",
                body["results"][0]["score"] > 0.9,
                f"top score {body['results'][0]['score']:.3f}",
                "similarity",
            )

    r = client.get(f"{BASE}/similarity/stats", headers=hk)
    ok = r.status_code == 200
    check("GET /similarity/stats", ok, f"{r.status_code}", "similarity")
    if ok:
        print(f"         {r.json()}")

    # -------------------------------------------------------------------- batch
    print("\nBatch (async)")
    items = [{"image_base64": b64, "image_id": f"batch-{i}"} for i in range(3)]
    r = client.post(f"{BASE}/batch", headers=hk, json={"task": "classification", "items": items})
    submitted = r.status_code == 202
    check("POST /batch returns 202", submitted, f"{r.status_code}", "batch")

    job_id = r.json().get("job_id") if submitted else None
    if job_id:
        final: dict[str, Any] = {}
        for _ in range(60):
            s = client.get(f"{BASE}/batch/{job_id}", headers=hk)
            if s.status_code == 200:
                final = s.json()
                if final.get("status") in ("completed", "failed"):
                    break
            time.sleep(1)
        check(
            "GET /batch/{job_id} reaches a terminal state",
            final.get("status") in ("completed", "failed"),
            f"status={final.get('status')} "
            f"completed={final.get('completed_items')}/{final.get('total_items')}",
            "batch",
        )

    r = client.post(f"{BASE}/batch", headers=hk, json={"task": "classification", "items": items})
    if r.status_code == 202:
        cancel_id = r.json()["job_id"]
        r = client.delete(f"{BASE}/batch/{cancel_id}", headers=hk)
        check(
            "DELETE /batch/{job_id}",
            r.status_code in (200, 202, 204, 409),
            f"{r.status_code}",
            "batch",
        )

    # ------------------------------------------------------------------- models
    print("\nModels")
    r = client.get(f"{BASE}/models", headers=hk)
    ok = r.status_code == 200
    check("GET /models", ok, f"{r.status_code}", "models")
    names = []
    if ok:
        names = [m["name"] for m in r.json()["models"]]
        print(f"         {len(names)} registered: {', '.join(sorted(set(names)))}")

    if names:
        r = client.get(f"{BASE}/models/{names[0]}", headers=hk)
        check("GET /models/{name}", r.status_code == 200, f"{r.status_code}", "models")

    r = client.get(f"{BASE}/models/does-not-exist", headers=hk)
    check("an unknown model returns 404", r.status_code == 404, f"{r.status_code}", "models")

    r = client.post(f"{BASE}/models/reload", headers=hk)
    check("POST /models/reload", r.status_code in (200, 403), f"{r.status_code}", "models")

    # ------------------------------------------------------------- input safety
    print("\nInput validation")
    r = client.post(f"{BASE}/classify", headers=hk, json={"image_base64": "not-base64!!"})
    check("malformed base64 is refused", r.status_code in (400, 422), f"{r.status_code}", "safety")

    r = client.post(
        f"{BASE}/classify",
        headers=hk,
        json={"image_base64": base64.b64encode(b"this is not an image").decode()},
    )
    check(
        "a non-image payload is refused", r.status_code in (400, 422), f"{r.status_code}", "safety"
    )

    r = client.post(
        f"{BASE}/classify",
        headers=hk,
        json={"image_url": "http://169.254.169.254/latest/meta-data/"},
    )
    check(
        "SSRF to link-local is refused", r.status_code in (400, 422), f"{r.status_code}", "safety"
    )

    r = client.post(f"{BASE}/classify", headers=hk, json={})
    check("an empty body is refused", r.status_code == 422, f"{r.status_code}", "safety")

    client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--target",
        choices=("gateway", "direct", "both"),
        default="gateway",
        help=(
            "gateway: through nginx on port 80, the default, which is what "
            "callers use. direct: the API's own port 8000, no gateway. "
            "both: run the suite against each in turn."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "Explicit root, without /api/v1, for a deployment that is not the "
            "local compose stack. Overrides --target."
        ),
    )
    parser.add_argument("--api-key", default="dev-key-pro")
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Image to send. Defaults to the first file in samples/.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write the results as JSON here.",
    )
    args = parser.parse_args()

    if args.base_url:
        targets = [("custom", args.base_url)]
    elif args.target == "both":
        targets = [("gateway", GATEWAY_URL), ("direct", DIRECT_URL)]
    elif args.target == "direct":
        targets = [("direct", DIRECT_URL)]
    else:
        targets = [("gateway", GATEWAY_URL)]

    image_path = args.image or next(iter(sorted((REPO_ROOT / "samples").glob("*.jpg"))), None)
    if image_path is None or not image_path.is_file():
        print("no image to send: pass --image or add one to samples/", file=sys.stderr)
        return 2
    print(f"image : {image_path.name}")

    for target, base_url in targets:
        try:
            run_suite(target, base_url, args.api_key, image_path, args.timeout)
        except httpx.ConnectError as exc:
            # Named rather than left as a traceback: the usual cause is that the
            # stack is not up, or that only one of the two ports is published.
            print(f"\ncannot reach {base_url}: {exc}", file=sys.stderr)
            results.append((target, "connection", f"reach {base_url}", False, str(exc)))

    # ------------------------------------------------------------------ summary
    passed = sum(1 for row in results if row[3])
    total = len(results)
    print(f"\n{'=' * 60}")
    if len(targets) > 1:
        for target, _ in targets:
            rows = [row for row in results if row[0] == target]
            print(f"{target:>8}: {sum(1 for row in rows if row[3])}/{len(rows)} passed")
    print(f"{passed}/{total} checks passed")
    failures = [(t, g, n, d) for t, g, n, ok, d in results if not ok]
    if failures:
        print("\nFailures:")
        for target, group, name, detail in failures:
            print(f"  [{target}/{group}] {name}: {detail}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "targets": dict(targets),
                    "passed": passed,
                    "total": total,
                    "checks": [
                        {"target": t, "group": g, "check": n, "ok": ok, "detail": d}
                        for t, g, n, ok, d in results
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"report: {args.report}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
