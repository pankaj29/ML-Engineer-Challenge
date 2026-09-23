"""A/B testing framework for comparing models.

Plain English:
    You have a new model and you think it is better. "Think" is the problem.
    This module decides whether the new model is *actually* better, or whether
    it just got lucky on the sample you tried it on.

Two distinct things live here:

1. **Offline comparison** — run both models over the same evaluation set and
   test whether the difference in accuracy is real. This is what you do
   *before* anything reaches a user.

2. **Online traffic splitting** — send a fraction of live requests to the
   challenger and compare outcomes. This is what you do *after* the offline
   test passes.

**The key statistical idea: use a paired test.** Both models see the exact
same images, so the comparison is naturally paired. The right test is
**McNemar's test**, which ignores every image both models got right and every
image both got wrong, and looks only at the disagreements: how many did A get
right and B get wrong, versus the reverse. Throwing away the agreements sounds
wasteful but is precisely what makes the test powerful — the agreements carry
no information about which model is better.

**Assignment must be deterministic.** A user hashed to the challenger must
*stay* on the challenger. Random assignment per request means one user sees
both models, which both ruins the statistics and produces visibly inconsistent
behaviour.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class ModelScores:
    """One model's results over an evaluation set."""

    name: str
    correct: list[bool] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    errors: int = 0

    @property
    def n(self) -> int:
        return len(self.correct)

    @property
    def accuracy(self) -> float:
        return float(np.mean(self.correct)) if self.correct else 0.0

    @property
    def mean_confidence(self) -> float:
        return float(np.mean(self.confidences)) if self.confidences else 0.0

    @property
    def p50_latency(self) -> float:
        return float(np.percentile(self.latencies_ms, 50)) if self.latencies_ms else 0.0

    @property
    def p95_latency(self) -> float:
        return float(np.percentile(self.latencies_ms, 95)) if self.latencies_ms else 0.0


@dataclass
class ABTestResult:
    """Verdict from comparing two models."""

    champion: str
    challenger: str
    n_samples: int

    champion_accuracy: float
    challenger_accuracy: float
    accuracy_delta: float

    test: str
    statistic: float
    p_value: float
    significant: bool
    confidence_interval: tuple[float, float]

    champion_p95_ms: float
    challenger_p95_ms: float
    latency_delta_ms: float

    winner: str
    recommendation: str
    detail: dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["confidence_interval"] = list(self.confidence_interval)
        return payload

    def summary(self) -> str:
        return (
            f"{self.challenger} vs {self.champion} on {self.n_samples} samples: "
            f"{self.champion_accuracy:.2%} -> {self.challenger_accuracy:.2%} "
            f"({self.accuracy_delta:+.2%}), p={self.p_value:.4f}. "
            f"Winner: {self.winner}."
        )


def mcnemar_test(
    champion_correct: Sequence[bool], challenger_correct: Sequence[bool]
) -> tuple[float, float, int, int]:
    """McNemar's test on paired correct/incorrect outcomes.

    Builds the two counts that matter:

        b = champion right, challenger wrong   (the challenger's regressions)
        c = champion wrong, challenger right   (the challenger's wins)

    Under the null hypothesis that the models are equally good, b and c should
    be about equal. The test asks how unlikely the observed imbalance is.

    For small counts (b + c < 25) the exact binomial test is used, because the
    chi-square approximation is unreliable there and will happily report
    significance that is not there.

    Returns:
        ``(statistic, p_value, b, c)``.
    """
    from scipy import stats

    champ = np.asarray(champion_correct, dtype=bool)
    chall = np.asarray(challenger_correct, dtype=bool)

    if champ.shape != chall.shape:
        raise ValueError(
            f"paired comparison needs equal-length results, got {champ.shape} and {chall.shape}"
        )

    b = int(np.sum(champ & ~chall))
    c = int(np.sum(~champ & chall))

    if b + c == 0:
        # The models agreed on every single sample.
        return 0.0, 1.0, b, c

    if b + c < 25:
        result = stats.binomtest(c, b + c, p=0.5, alternative="two-sided")
        return float(c), float(result.pvalue), b, c

    # Chi-square with Yates' continuity correction, the standard form.
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(stats.chi2.sf(statistic, df=1))
    return float(statistic), p_value, b, c


def proportion_confidence_interval(
    successes_a: int, n_a: int, successes_b: int, n_b: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Confidence interval for the difference between two accuracy rates.

    The interval is usually more informative than the p-value. "The challenger
    is 2.1% better, 95% CI [0.4%, 3.8%]" tells you both that the improvement
    is real and roughly how big it is. "p = 0.03" tells you only the first.

    An interval that spans zero means the data cannot rule out "no difference".
    """
    from scipy import stats

    if n_a == 0 or n_b == 0:
        return (0.0, 0.0)

    p_a, p_b = successes_a / n_a, successes_b / n_b
    diff = p_b - p_a
    se = float(np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b))
    z = float(stats.norm.ppf(1 - (1 - confidence) / 2))
    return (round(diff - z * se, 6), round(diff + z * se, 6))


def compare(
    champion: ModelScores,
    challenger: ModelScores,
    *,
    alpha: float = 0.05,
    min_improvement: float = 0.005,
    max_latency_regression_ms: float = 50.0,
) -> ABTestResult:
    """Compare two models and recommend whether to promote the challenger.

    The recommendation weighs three things, because a model decision is never
    purely about accuracy:

    * Is the accuracy difference statistically significant?
    * Is it large enough to be worth a deployment (``min_improvement``)?
    * Did latency get materially worse (``max_latency_regression_ms``)?

    A challenger that is 0.2% more accurate and 300 ms slower is not an
    improvement, and this function says so.

    Args:
        min_improvement: Smallest accuracy gain worth shipping. Defaults to
            0.5 percentage points.
        max_latency_regression_ms: p95 latency increase that makes a win not
            worth taking.
    """
    if champion.n != challenger.n:
        raise ValueError(
            f"A/B comparison requires both models to be evaluated on the same samples "
            f"({champion.n} vs {challenger.n}). An unpaired comparison would be invalid."
        )

    statistic, p_value, b, c = mcnemar_test(champion.correct, challenger.correct)

    champ_acc = champion.accuracy
    chall_acc = challenger.accuracy
    delta = chall_acc - champ_acc

    ci = proportion_confidence_interval(
        int(np.sum(champion.correct)),
        champion.n,
        int(np.sum(challenger.correct)),
        challenger.n,
    )

    significant = bool(p_value < alpha)
    latency_delta = challenger.p95_latency - champion.p95_latency
    latency_acceptable = latency_delta <= max_latency_regression_ms

    if not significant:
        winner = "tie"
        recommendation = (
            f"No significant difference (p={p_value:.4f}). Keep {champion.name}. "
            f"The challenger disagreed on {b + c} of {champion.n} samples, winning {c} "
            f"and losing {b} — too close to call. "
            + (
                f"Collect more evaluation data: at this effect size you would need "
                f"roughly {_required_sample_size(champ_acc, delta):,} samples to detect it."
                if abs(delta) > 0
                else "The models are effectively identical on this set."
            )
        )
    elif delta > 0 and delta >= min_improvement and latency_acceptable:
        winner = challenger.name
        recommendation = (
            f"Promote {challenger.name}. Accuracy improved by {delta:+.2%} "
            f"(95% CI [{ci[0]:+.2%}, {ci[1]:+.2%}], p={p_value:.4f}) with acceptable "
            f"latency ({latency_delta:+.1f} ms p95). Roll out gradually and watch "
            "for drift before retiring the champion."
        )
    elif delta > 0 and not latency_acceptable:
        winner = "champion"
        recommendation = (
            f"Do NOT promote. {challenger.name} is {delta:+.2%} more accurate but "
            f"{latency_delta:+.1f} ms slower at p95, exceeding the "
            f"{max_latency_regression_ms} ms budget. Optimise it (quantize or export "
            "to a faster runtime) and re-test."
        )
    elif delta > 0:
        winner = "champion"
        recommendation = (
            f"Do NOT promote. The {delta:+.2%} gain is statistically real but below the "
            f"{min_improvement:.2%} threshold that justifies a deployment. Keep "
            f"{champion.name}."
        )
    else:
        winner = champion.name
        recommendation = (
            f"Do NOT promote — {challenger.name} is significantly WORSE by {abs(delta):.2%} "
            f"(p={p_value:.4f}). It lost on {b} samples the champion got right. "
            "Investigate before retraining: check for a preprocessing mismatch, which is "
            "the most common cause of an unexpectedly worse challenger."
        )

    return ABTestResult(
        champion=champion.name,
        challenger=challenger.name,
        n_samples=champion.n,
        champion_accuracy=round(champ_acc, 6),
        challenger_accuracy=round(chall_acc, 6),
        accuracy_delta=round(delta, 6),
        test="mcnemar_exact" if (b + c) < 25 else "mcnemar_chi2",
        statistic=round(statistic, 6),
        p_value=float(p_value),
        significant=significant,
        confidence_interval=ci,
        champion_p95_ms=round(champion.p95_latency, 2),
        challenger_p95_ms=round(challenger.p95_latency, 2),
        latency_delta_ms=round(latency_delta, 2),
        winner=winner,
        recommendation=recommendation,
        detail={
            "champion_only_correct": b,
            "challenger_only_correct": c,
            "both_correct": int(
                np.sum(np.asarray(champion.correct) & np.asarray(challenger.correct))
            ),
            "both_wrong": int(
                np.sum(~np.asarray(champion.correct) & ~np.asarray(challenger.correct))
            ),
            "champion_mean_confidence": round(champion.mean_confidence, 4),
            "challenger_mean_confidence": round(challenger.mean_confidence, 4),
            "champion_errors": champion.errors,
            "challenger_errors": challenger.errors,
            "alpha": alpha,
            "min_improvement": min_improvement,
        },
    )


def _required_sample_size(
    baseline: float, delta: float, power: float = 0.8, alpha: float = 0.05
) -> int:
    """Roughly how many samples are needed to detect an effect of this size.

    Answers the question people actually ask after an inconclusive test: "how
    much more data do I need?" Uses the standard two-proportion formula.
    """
    from scipy import stats

    if delta == 0:
        return 0
    p1 = baseline
    p2 = min(max(baseline + delta, 1e-6), 1 - 1e-6)
    p_bar = (p1 + p2) / 2

    z_alpha = float(stats.norm.ppf(1 - alpha / 2))
    z_beta = float(stats.norm.ppf(power))

    numerator = (
        z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    return int(np.ceil(numerator / (delta**2)))


# ---------------------------------------------------------------------------
# Online traffic splitting
# ---------------------------------------------------------------------------
@dataclass
class TrafficSplit:
    """Deterministic assignment of callers to model variants.

    Assignment is by hashing the user id, so the same user always lands on the
    same variant for the life of the experiment. See the module docstring for
    why random-per-request assignment is wrong.

    The ``salt`` matters: it should be the experiment name. Without it, every
    experiment would assign the same users to the challenger, and effects from
    different experiments would pile up on one unlucky group.
    """

    champion: str
    challenger: str
    challenger_percent: float = 10.0
    salt: str = "experiment-1"
    enabled: bool = True

    def variant_for(self, user_id: str) -> str:
        """Return the model this user should be served by."""
        if not self.enabled or self.challenger_percent <= 0:
            return self.champion
        if self.challenger_percent >= 100:
            return self.challenger

        digest = hashlib.sha256(f"{self.salt}:{user_id}".encode()).hexdigest()
        # Take 8 hex chars and map onto 0-100. SHA-256 is uniform, so the
        # split lands within a fraction of a percent of the target.
        bucket = int(digest[:8], 16) / 0xFFFFFFFF * 100.0
        return self.challenger if bucket < self.challenger_percent else self.champion

    def describe(self) -> dict[str, Any]:
        return {
            "champion": self.champion,
            "challenger": self.challenger,
            "challenger_percent": self.challenger_percent,
            "enabled": self.enabled,
            "salt": self.salt,
        }


def verify_split_balance(split: TrafficSplit, sample_users: int = 100_000) -> dict[str, float]:
    """Check that the hash actually produces the intended split.

    Worth running once per experiment: a bug here silently invalidates the
    whole result, and the symptom (a slightly odd sample size) is easy to miss.
    """
    counts = {split.champion: 0, split.challenger: 0}
    for i in range(sample_users):
        counts[split.variant_for(f"user_{i}")] += 1
    return {
        "target_challenger_percent": split.challenger_percent,
        "actual_challenger_percent": round(counts[split.challenger] / sample_users * 100, 3),
        "sample_size": sample_users,
    }


# ---------------------------------------------------------------------------
# Offline evaluation harness
# ---------------------------------------------------------------------------
def evaluate_models(
    predict_fns: dict[str, Callable[[bytes], tuple[int, float, float]]],
    samples: list[tuple[bytes, int]],
) -> dict[str, ModelScores]:
    """Run several models over the same samples.

    Args:
        predict_fns: ``{model_name: fn}`` where each fn takes image bytes and
            returns ``(predicted_class, confidence, latency_ms)``.
        samples: ``[(image_bytes, true_label), ...]``.

    Returns:
        ``{model_name: ModelScores}``, all evaluated on identical samples so
        the paired test in :func:`compare` is valid.
    """
    scores = {name: ModelScores(name=name) for name in predict_fns}

    for image_bytes, true_label in samples:
        for name, fn in predict_fns.items():
            try:
                predicted, confidence, latency = fn(image_bytes)
                scores[name].correct.append(predicted == true_label)
                scores[name].confidences.append(confidence)
                scores[name].latencies_ms.append(latency)
            except Exception:
                # A failure counts as wrong, not as a missing sample. Dropping
                # it would leave the models with unequal sample counts and
                # invalidate the paired comparison.
                scores[name].correct.append(False)
                scores[name].confidences.append(0.0)
                scores[name].latencies_ms.append(0.0)
                scores[name].errors += 1

    return scores


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="A/B compare two registered models.")
    parser.add_argument("--champion", required=True, help="Current model, as name:version.")
    parser.add_argument("--challenger", required=True, help="Candidate model, as name:version.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-improvement", type=float, default=0.005)
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "benchmarks" / "reports" / "ab_test.json"
    )
    args = parser.parse_args()

    import time

    from api.services.model_service import ModelService
    from api.utils.image_processing import preprocess
    from models.training.dataset import TinyImageNetTrain, TinyImageNetVal, find_dataset_root

    service = ModelService()

    def build_predictor(spec: str) -> Callable[[bytes], tuple[int, float, float]]:
        name, _, version = spec.partition(":")
        entry = service.resolve(
            next(e.task for e in service.list_entries() if e.name == name),
            name,
            version or "latest",
        )
        loaded = service.load(entry)
        cfg = loaded.preprocess_config

        def predict(image_bytes: bytes) -> tuple[int, float, float]:
            arr = preprocess(image_bytes, cfg).array
            start = time.perf_counter()
            outputs = loaded.runtime.infer(arr)
            latency = (time.perf_counter() - start) * 1000
            logits = outputs[0][0]
            exp = np.exp(logits - logits.max())
            probs = exp / exp.sum()
            return int(probs.argmax()), float(probs.max()), latency

        return predict

    root = find_dataset_root(args.data_dir)
    train = TinyImageNetTrain(root)
    val = TinyImageNetVal(root, train.class_to_idx)
    samples = [(path.read_bytes(), label) for path, label in val.samples[: args.samples]]
    print(f"evaluating on {len(samples)} samples")

    scores = evaluate_models(
        {
            args.champion: build_predictor(args.champion),
            args.challenger: build_predictor(args.challenger),
        },
        samples,
    )

    result = compare(
        scores[args.champion],
        scores[args.challenger],
        alpha=args.alpha,
        min_improvement=args.min_improvement,
    )

    print("\n" + result.summary())
    print("\n" + result.recommendation)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
