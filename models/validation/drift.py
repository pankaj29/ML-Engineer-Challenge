"""Model drift detection using statistical tests.

Plain English:
    A model is trained on the world as it was. The world then moves on, and
    the model gets quietly worse without anything crashing. That silent decay
    is **drift**, and it is the most common way ML systems fail in production.

    You usually cannot measure accuracy live, because nobody tells you the
    right answer. So instead you watch the *distributions* and ask: does the
    data coming in today look like the data the model was trained on? Do
    today's predictions look like yesterday's?

Three kinds of drift, and the test used for each:

* **Data drift** (the inputs changed) — a camera was replaced, so images are
  now brighter. Detected with the **Kolmogorov-Smirnov test** on continuous
  features: it measures the largest gap between two cumulative distributions.

* **Prediction drift** (the outputs changed) — the model used to say "cat" 8%
  of the time and now says it 40%. Detected with the **chi-square test** on
  the category counts.

* **Confidence drift** (certainty changed) — the model is still saying "cat",
  but with 0.5 confidence where it used to say 0.95. Usually the earliest
  warning of all, and detected with KS plus **Population Stability Index**.

**On p-values.** A p-value answers "if nothing really changed, how surprising
is this data?" Small p means surprising, so something probably did change. The
trap is sample size: with 100,000 requests, a difference far too small to
matter will still produce p < 0.001. That is why every test here reports an
**effect size** alongside the p-value, and the verdict requires *both*
statistical significance and a practically meaningful effect. Alerting on
p-values alone produces a monitor everyone learns to ignore.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class DriftSeverity(str, Enum):
    """How worried to be."""

    NONE = "none"  # nothing detected
    LOW = "low"  # detectable, not actionable
    MODERATE = "moderate"  # investigate
    HIGH = "high"  # act now


# Population Stability Index thresholds. These are the long-standing
# conventions from credit-risk modelling, where PSI originated.
PSI_THRESHOLDS = {"low": 0.1, "moderate": 0.2, "high": 0.25}


@dataclass
class DriftResult:
    """Outcome of one drift test."""

    test: str
    feature: str
    statistic: float
    p_value: float
    effect_size: float
    effect_metric: str
    drifted: bool
    severity: DriftSeverity
    reference_size: int
    current_size: int
    detail: dict[str, Any] = field(default_factory=dict)
    interpretation: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["severity"] = self.severity.value
        return payload


def _severity_from_effect(effect: float, thresholds: tuple[float, float, float]) -> DriftSeverity:
    """Map an effect size onto a severity using three ascending cut-offs."""
    low, moderate, high = thresholds
    if effect >= high:
        return DriftSeverity.HIGH
    if effect >= moderate:
        return DriftSeverity.MODERATE
    if effect >= low:
        return DriftSeverity.LOW
    return DriftSeverity.NONE


# ---------------------------------------------------------------------------
# Continuous features: KS test
# ---------------------------------------------------------------------------
def ks_drift(
    reference: Sequence[float],
    current: Sequence[float],
    *,
    feature: str = "value",
    alpha: float = 0.05,
    min_effect: float = 0.1,
) -> DriftResult:
    """Two-sample Kolmogorov-Smirnov test for continuous values.

    How it works, intuitively: draw the cumulative distribution of both
    samples and measure the largest vertical gap between the two curves. That
    gap is the statistic, and it doubles as the effect size — it is already on
    a 0-to-1 scale where 0 means identical and 1 means completely separated.

    The KS test makes no assumption about the shape of the distribution, which
    is why it suits confidence scores (typically heavily skewed towards 1.0)
    where a t-test's normality assumption would be wrong.

    Args:
        reference: Baseline sample, e.g. confidences at model release.
        current: Recent sample.
        alpha: Significance level.
        min_effect: Minimum KS statistic to treat as practically meaningful.
            Prevents alerting on a tiny-but-significant shift in a big sample.
    """
    from scipy import stats

    ref = np.asarray([x for x in reference if x is not None], dtype=np.float64)
    cur = np.asarray([x for x in current if x is not None], dtype=np.float64)

    if len(ref) < 20 or len(cur) < 20:
        return DriftResult(
            test="kolmogorov_smirnov",
            feature=feature,
            statistic=0.0,
            p_value=1.0,
            effect_size=0.0,
            effect_metric="ks_statistic",
            drifted=False,
            severity=DriftSeverity.NONE,
            reference_size=len(ref),
            current_size=len(cur),
            interpretation=(
                "Not enough data to test: at least 20 samples are needed on each "
                f"side (have {len(ref)} reference, {len(cur)} current)."
            ),
        )

    statistic, p_value = stats.ks_2samp(ref, cur)
    significant = bool(p_value < alpha)
    meaningful = bool(statistic >= min_effect)
    drifted = significant and meaningful

    severity = (
        _severity_from_effect(float(statistic), (0.1, 0.2, 0.3))
        if significant
        else DriftSeverity.NONE
    )

    if drifted:
        direction = "higher" if float(np.mean(cur)) > float(np.mean(ref)) else "lower"
        interpretation = (
            f"{feature} has drifted: the distribution shifted {direction} "
            f"(mean {np.mean(ref):.3f} -> {np.mean(cur):.3f}). "
            f"Largest CDF gap {statistic:.3f}, p={p_value:.2e}."
        )
    elif significant and not meaningful:
        interpretation = (
            f"{feature} shows a statistically significant but tiny difference "
            f"(KS {statistic:.3f} < {min_effect} threshold). This is sample-size "
            "sensitivity, not a real change. No action needed."
        )
    else:
        interpretation = (
            f"{feature} shows no meaningful drift (KS {statistic:.3f}, p={p_value:.2f})."
        )

    return DriftResult(
        test="kolmogorov_smirnov",
        feature=feature,
        statistic=round(float(statistic), 6),
        p_value=float(p_value),
        effect_size=round(float(statistic), 6),
        effect_metric="ks_statistic",
        drifted=drifted,
        severity=severity,
        reference_size=len(ref),
        current_size=len(cur),
        detail={
            "reference_mean": round(float(np.mean(ref)), 4),
            "current_mean": round(float(np.mean(cur)), 4),
            "reference_std": round(float(np.std(ref)), 4),
            "current_std": round(float(np.std(cur)), 4),
            "reference_median": round(float(np.median(ref)), 4),
            "current_median": round(float(np.median(cur)), 4),
        },
        interpretation=interpretation,
    )


# ---------------------------------------------------------------------------
# Categorical features: chi-square
# ---------------------------------------------------------------------------
def chi_square_drift(
    reference: Sequence[str],
    current: Sequence[str],
    *,
    feature: str = "predicted_label",
    alpha: float = 0.05,
    min_effect: float = 0.1,
) -> DriftResult:
    """Chi-square test on category frequencies.

    Compares how often each category appears now against how often it appeared
    in the reference period. Effect size is **Cramér's V**, which rescales the
    chi-square statistic onto 0-to-1 so it can be compared across features with
    different numbers of categories.

    Categories present in one sample but not the other are included with a
    count of zero, because a class that has *stopped* appearing entirely is
    exactly the kind of drift worth catching.
    """
    from scipy import stats

    ref_counts = Counter(str(x) for x in reference if x is not None)
    cur_counts = Counter(str(x) for x in current if x is not None)

    ref_total = sum(ref_counts.values())
    cur_total = sum(cur_counts.values())

    if ref_total < 20 or cur_total < 20:
        return DriftResult(
            test="chi_square",
            feature=feature,
            statistic=0.0,
            p_value=1.0,
            effect_size=0.0,
            effect_metric="cramers_v",
            drifted=False,
            severity=DriftSeverity.NONE,
            reference_size=ref_total,
            current_size=cur_total,
            interpretation=(
                f"Not enough data to test (have {ref_total} reference, {cur_total} current; "
                "20 needed on each side)."
            ),
        )

    categories = sorted(set(ref_counts) | set(cur_counts))
    observed = np.array(
        [[ref_counts.get(c, 0) for c in categories], [cur_counts.get(c, 0) for c in categories]],
        dtype=np.float64,
    )

    # Drop categories absent from both samples: an all-zero column makes the
    # expected frequency zero and the test undefined.
    keep = observed.sum(axis=0) > 0
    observed = observed[:, keep]
    categories = [c for c, k in zip(categories, keep, strict=False) if k]

    if observed.shape[1] < 2:
        return DriftResult(
            test="chi_square",
            feature=feature,
            statistic=0.0,
            p_value=1.0,
            effect_size=0.0,
            effect_metric="cramers_v",
            drifted=False,
            severity=DriftSeverity.NONE,
            reference_size=ref_total,
            current_size=cur_total,
            interpretation="Only one category present; there is nothing to compare.",
        )

    statistic, p_value, _, _ = stats.chi2_contingency(observed)

    # Cramér's V = sqrt(chi2 / (n * (min(rows, cols) - 1))).
    n = observed.sum()
    min_dim = min(observed.shape) - 1
    cramers_v = float(np.sqrt(statistic / (n * min_dim))) if n and min_dim else 0.0

    significant = bool(p_value < alpha)
    meaningful = bool(cramers_v >= min_effect)
    drifted = significant and meaningful

    # Which categories moved the most, in percentage-point terms.
    shifts = []
    for idx, category in enumerate(categories):
        # float() casts away NumPy scalar types, which json.dumps cannot encode.
        ref_pct = float(observed[0, idx]) / ref_total * 100
        cur_pct = float(observed[1, idx]) / cur_total * 100
        shifts.append((category, round(ref_pct, 2), round(cur_pct, 2), round(cur_pct - ref_pct, 2)))
    shifts.sort(key=lambda s: abs(s[3]), reverse=True)
    top_shifts = shifts[:5]

    if drifted:
        biggest = top_shifts[0]
        interpretation = (
            f"Prediction mix for {feature} has drifted (Cramer's V {cramers_v:.3f}, "
            f"p={p_value:.2e}). Largest change: '{biggest[0]}' went from "
            f"{biggest[1]}% to {biggest[2]}% ({biggest[3]:+.2f} points)."
        )
    elif significant and not meaningful:
        interpretation = (
            f"Statistically significant but negligible shift in {feature} "
            f"(Cramer's V {cramers_v:.3f} < {min_effect}). No action needed."
        )
    else:
        interpretation = (
            f"No meaningful shift in {feature} (Cramer's V {cramers_v:.3f}, p={p_value:.2f})."
        )

    return DriftResult(
        test="chi_square",
        feature=feature,
        statistic=round(float(statistic), 4),
        p_value=float(p_value),
        effect_size=round(cramers_v, 6),
        effect_metric="cramers_v",
        drifted=drifted,
        severity=(
            _severity_from_effect(cramers_v, (0.1, 0.2, 0.35))
            if significant
            else DriftSeverity.NONE
        ),
        reference_size=ref_total,
        current_size=cur_total,
        detail={
            "num_categories": len(categories),
            "top_shifts": [
                {"category": c, "reference_pct": r, "current_pct": u, "change_pct": d}
                for c, r, u, d in top_shifts
            ],
            "new_categories": sorted(set(cur_counts) - set(ref_counts))[:10],
            "vanished_categories": sorted(set(ref_counts) - set(cur_counts))[:10],
        },
        interpretation=interpretation,
    )


# ---------------------------------------------------------------------------
# Population Stability Index
# ---------------------------------------------------------------------------
def population_stability_index(
    reference: Sequence[float],
    current: Sequence[float],
    *,
    feature: str = "confidence",
    bins: int = 10,
) -> DriftResult:
    """Population Stability Index between two continuous distributions.

    PSI chops the reference distribution into equal-frequency buckets, then
    asks how much of the population moved between buckets. It produces a
    single number with industry-standard cut-offs:

        < 0.10  stable
        0.10 - 0.25  moderate shift, worth investigating
        > 0.25  major shift, act

    Unlike KS it has no p-value — it is purely an effect size, which makes it
    immune to the large-sample problem described in the module docstring. That
    is exactly why it pairs well with KS rather than replacing it.
    """
    ref = np.asarray([x for x in reference if x is not None], dtype=np.float64)
    cur = np.asarray([x for x in current if x is not None], dtype=np.float64)

    if len(ref) < 20 or len(cur) < 20:
        return DriftResult(
            test="psi",
            feature=feature,
            statistic=0.0,
            p_value=1.0,
            effect_size=0.0,
            effect_metric="psi",
            drifted=False,
            severity=DriftSeverity.NONE,
            reference_size=len(ref),
            current_size=len(cur),
            interpretation="Not enough data to compute PSI (20 samples needed on each side).",
        )

    # Quantile edges from the reference, so buckets hold equal reference mass.
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        # Nearly constant reference: fall back to a fixed-width split.
        edges = np.linspace(min(ref.min(), cur.min()), max(ref.max(), cur.max()), bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf

    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)

    ref_pct = ref_hist / max(ref_hist.sum(), 1)
    cur_pct = cur_hist / max(cur_hist.sum(), 1)

    # An empty bucket would make the log term infinite. Substituting a tiny
    # floor keeps the sum finite while still registering the bucket as moved.
    epsilon = 1e-6
    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    severity = _severity_from_effect(
        psi, (PSI_THRESHOLDS["low"], PSI_THRESHOLDS["moderate"], PSI_THRESHOLDS["high"])
    )
    drifted = psi >= PSI_THRESHOLDS["moderate"]

    if psi < PSI_THRESHOLDS["low"]:
        verdict = "stable"
    elif psi < PSI_THRESHOLDS["moderate"]:
        verdict = "minor shift, keep watching"
    elif psi < PSI_THRESHOLDS["high"]:
        verdict = "moderate shift, investigate"
    else:
        verdict = "major shift, action required"

    return DriftResult(
        test="psi",
        feature=feature,
        statistic=round(psi, 6),
        p_value=1.0,  # PSI has no p-value by construction
        effect_size=round(psi, 6),
        effect_metric="psi",
        drifted=drifted,
        severity=severity,
        reference_size=len(ref),
        current_size=len(cur),
        detail={
            "bins": len(ref_pct),
            "reference_distribution": [round(float(x), 4) for x in ref_pct],
            "current_distribution": [round(float(x), 4) for x in cur_pct],
        },
        interpretation=f"PSI for {feature} is {psi:.4f} ({verdict}).",
    )


# ---------------------------------------------------------------------------
# Image statistics (input drift)
# ---------------------------------------------------------------------------
def image_statistics(image_bytes: bytes) -> dict[str, float]:
    """Summarise an image as a handful of numbers for drift monitoring.

    Storing user images to compare later is a privacy and storage problem.
    These few statistics capture the shifts that actually matter in practice —
    a new camera, a change in lighting, a different upload pipeline — without
    retaining anything identifiable.
    """
    import io

    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0

    grey = arr.mean(axis=2)
    return {
        "mean_brightness": float(arr.mean()),
        "std_brightness": float(arr.std()),
        "mean_red": float(arr[:, :, 0].mean()),
        "mean_green": float(arr[:, :, 1].mean()),
        "mean_blue": float(arr[:, :, 2].mean()),
        "contrast": float(grey.max() - grey.min()),
        # Edge density: a rough proxy for how much detail the image holds.
        # Blurry or heavily compressed uploads drop sharply on this.
        "edge_density": float(
            np.abs(np.diff(grey, axis=0)).mean() + np.abs(np.diff(grey, axis=1)).mean()
        ),
        "aspect_ratio": float(arr.shape[1] / max(arr.shape[0], 1)),
        "pixels": float(arr.shape[0] * arr.shape[1]),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
@dataclass
class DriftReport:
    """A full drift assessment across several signals."""

    model: str
    generated_at: str
    results: list[dict[str, Any]]
    overall_severity: str
    drifted: bool
    summary: str
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_drift(
    *,
    model: str,
    reference_confidences: Sequence[float] | None = None,
    current_confidences: Sequence[float] | None = None,
    reference_labels: Sequence[str] | None = None,
    current_labels: Sequence[str] | None = None,
    reference_features: dict[str, Sequence[float]] | None = None,
    current_features: dict[str, Sequence[float]] | None = None,
    alpha: float = 0.05,
) -> DriftReport:
    """Run every applicable drift test and combine them into one verdict.

    Args:
        model: ``name:version`` being assessed.
        reference_*: Baseline sample, usually from the week after release.
        current_*: Recent sample, usually the last 24 hours.

    Returns:
        A :class:`DriftReport` whose ``overall_severity`` is the worst of the
        individual results — one badly drifted signal should not be averaged
        away by several stable ones.
    """
    results: list[DriftResult] = []

    # `len(x)` rather than truthiness: a NumPy array raises ValueError when
    # used in a boolean context.
    def _has(seq: Sequence[Any] | None) -> bool:
        return seq is not None and len(seq) > 0

    if _has(reference_confidences) and _has(current_confidences):
        results.append(
            ks_drift(reference_confidences, current_confidences, feature="confidence", alpha=alpha)
        )
        results.append(
            population_stability_index(
                reference_confidences, current_confidences, feature="confidence"
            )
        )

    if _has(reference_labels) and _has(current_labels):
        results.append(
            chi_square_drift(
                reference_labels, current_labels, feature="predicted_label", alpha=alpha
            )
        )

    if reference_features is not None and current_features is not None:
        for name in sorted(set(reference_features) & set(current_features)):
            results.append(
                ks_drift(
                    reference_features[name], current_features[name], feature=name, alpha=alpha
                )
            )

    order = [DriftSeverity.NONE, DriftSeverity.LOW, DriftSeverity.MODERATE, DriftSeverity.HIGH]
    worst = max((r.severity for r in results), key=order.index, default=DriftSeverity.NONE)
    drifted_results = [r for r in results if r.drifted]

    if not results:
        summary = "No drift tests could be run: no comparable data was supplied."
    elif not drifted_results:
        summary = f"No meaningful drift detected across {len(results)} test(s)."
    else:
        names = ", ".join(sorted({r.feature for r in drifted_results}))
        summary = (
            f"Drift detected in {len(drifted_results)} of {len(results)} test(s). "
            f"Affected signals: {names}. Overall severity: {worst.value}."
        )

    recommendations: list[str] = []
    if worst in (DriftSeverity.MODERATE, DriftSeverity.HIGH):
        recommendations.append(
            "Pull a sample of recent inputs and label them by hand to measure real accuracy. "
            "Distribution drift is a warning sign, not proof the model got worse."
        )
        recommendations.append(
            "Compare against the deployment timeline: a step change that lines up with a "
            "release usually means a preprocessing or client change, not model decay."
        )
    if any(r.feature == "confidence" and r.drifted for r in results):
        recommendations.append(
            "Falling confidence with a stable prediction mix usually means the inputs have "
            "moved away from the training distribution. Retraining on recent data is the fix."
        )
    if any(r.feature == "predicted_label" and r.drifted for r in results):
        recommendations.append(
            "A shifted prediction mix can be a genuine change in traffic rather than a model "
            "problem. Confirm with whoever owns the upstream product before retraining."
        )
    if worst == DriftSeverity.HIGH:
        recommendations.append(
            "Severity is high: consider pinning clients to the previous model version while "
            "investigating, using the model_version parameter."
        )

    return DriftReport(
        model=model,
        generated_at=datetime.now(UTC).isoformat(),
        results=[r.to_dict() for r in results],
        overall_severity=worst.value,
        drifted=bool(drifted_results),
        summary=summary,
        recommendations=recommendations,
    )


async def detect_drift_from_database(
    model_name: str,
    model_version: str | None = None,
    *,
    reference_days: int = 30,
    current_days: int = 1,
) -> DriftReport:
    """Run drift detection against the live inference log.

    Compares the most recent ``current_days`` of predictions against the
    ``reference_days`` window that preceded them.
    """
    from datetime import timedelta

    from api.services.db_service import get_db_service

    db = get_db_service()
    now = datetime.now(UTC)

    current = await db.recent_predictions(
        model_name, model_version, since=now - timedelta(days=current_days)
    )
    reference_all = await db.recent_predictions(
        model_name, model_version, since=now - timedelta(days=reference_days + current_days)
    )
    cutoff = now - timedelta(days=current_days)
    reference = [r for r in reference_all if r["created_at"] < cutoff]

    return detect_drift(
        model=f"{model_name}:{model_version or 'latest'}",
        reference_confidences=[r["confidence"] for r in reference if r["confidence"] is not None],
        current_confidences=[r["confidence"] for r in current if r["confidence"] is not None],
        reference_labels=[r["label"] for r in reference if r["label"]],
        current_labels=[r["label"] for r in current if r["label"]],
    )


def _image_files(directory: Path, limit: int, offset: int = 0) -> list[Path]:
    files = sorted(
        f for f in Path(directory).rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    return files[offset : offset + limit]


def detect_drift_between_dirs(
    model_name: str,
    reference: list[Path],
    current: list[Path],
    *,
    model_version: str | None = None,
) -> DriftReport:
    """Drift between two image sets, run through the model's serving path.

    The inference log only has a reference window once a deployment has a
    month of traffic. This runs the same tests on two sets of images instead,
    which is how the detector is exercised before that history exists.
    """
    from api.models.schemas import TaskType
    from api.services.model_service import ModelService
    from api.utils.image_processing import preprocess

    service = ModelService()
    entry = service.resolve(TaskType.CLASSIFICATION, model_name, model_version)
    loaded = service.load(entry)

    def observe(paths: list[Path]) -> tuple[list[float], list[str], dict[str, list[float]]]:
        confidences: list[float] = []
        labels: list[str] = []
        features: dict[str, list[float]] = {}
        for path in paths:
            data = path.read_bytes()
            logits = loaded.runtime.infer(preprocess(data, loaded.preprocess_config).array)[0][0]
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            confidences.append(float(probs.max()))
            labels.append(loaded.label_for(int(probs.argmax())))
            for key, value in image_statistics(data).items():
                features.setdefault(key, []).append(value)
        return confidences, labels, features

    ref_conf, ref_labels, ref_features = observe(reference)
    cur_conf, cur_labels, cur_features = observe(current)
    return detect_drift(
        model=entry.key,
        reference_confidences=ref_conf,
        current_confidences=cur_conf,
        reference_labels=ref_labels,
        current_labels=cur_labels,
        reference_features=ref_features,
        current_features=cur_features,
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run drift detection against the inference log.")
    parser.add_argument("--model", required=True, help="Model name to analyse.")
    parser.add_argument("--version", default=None)
    parser.add_argument("--reference-days", type=int, default=30)
    parser.add_argument("--current-days", type=int, default=1)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="Compare two image directories instead of reading the inference log.",
    )
    parser.add_argument("--current-dir", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=300, help="Images per directory.")
    parser.add_argument(
        "--current-offset",
        type=int,
        default=0,
        help="Skip this many current images; with the same directory, a disjoint control sample.",
    )
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "benchmarks" / "reports" / "drift_report.json"
    )
    args = parser.parse_args()

    if args.reference_dir or args.current_dir:
        if not (args.reference_dir and args.current_dir):
            print("error: --reference-dir and --current-dir go together", file=sys.stderr)
            return 2
        reference = _image_files(args.reference_dir, args.samples)
        current = _image_files(args.current_dir, args.samples, args.current_offset)
        if not reference or not current:
            print("error: no images found in one of the directories", file=sys.stderr)
            return 2
        report = detect_drift_between_dirs(
            args.model, reference, current, model_version=args.version
        )
    else:
        import asyncio

        from api.services.db_service import get_db_service

        async def run() -> DriftReport:
            await get_db_service().connect()
            return await detect_drift_from_database(
                args.model,
                args.version,
                reference_days=args.reference_days,
                current_days=args.current_days,
            )

        report = asyncio.run(run())

    print(f"model    : {report.model}")
    print(f"severity : {report.overall_severity}")
    print(f"summary  : {report.summary}\n")
    for result in report.results:
        flag = "DRIFT" if result["drifted"] else "  ok "
        print(f"[{flag}] {result['test']:<20} {result['feature']:<18} {result['interpretation']}")
    if report.recommendations:
        print("\nrecommendations:")
        for rec in report.recommendations:
            print(f"  - {rec}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
