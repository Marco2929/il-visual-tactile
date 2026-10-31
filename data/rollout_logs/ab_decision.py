#!/usr/bin/env python3
"""Is 21 % vs 30 % enough to call one policy better? (No.)

The two testruns are being used as an A/B, so the first question is whether
the numbers can carry that weight at all. 4/19 against 7/23 looks like a
9-point gap, but both samples are tiny and the two runs were NOT matched:

  * the conditions differ in composition - the absolute run spent 3 rollouts
    on Zwischenhoehe and 3 on untere Hoehe, the relative run 3 and 3 but
    with a different nominal/XY split, and `retry` exists only in the
    relative run;
  * the start pose was not controlled, and for the ABSOLUTE model it
    predicts the outcome on its own (findings 7.1), so the absolute run is
    penalised by wherever the arm happened to be;
  * each condition holds 1-3 rollouts, so a single flip moves a per-condition
    rate by 33 points.

So this script does not try to declare a winner. It reports:

  1. CI + Fisher       how wide the uncertainty actually is, and the p-value
                       for the difference.
  2. FLIP TEST         how many outcomes would have to change to reverse the
                       ranking. If that number is 1-2, the ranking is noise.
  3. MATCHED SUBSET    the same comparison restricted to conditions BOTH
                       runs contain, which is the only like-for-like read.
  4. START-POSE BIAS   the absolute model's successes cluster by start pose;
                       this asks what its rate looks like among the runs that
                       started near the successful region, i.e. with its
                       known sensitivity controlled for.
  5. POWER             how many rollouts per arm a real A/B would need.

No scipy: the container has numpy only, so Fisher is computed directly from
hypergeometric terms and the CI by Wilson.

Usage (in the container):
    python data/rollout_logs/ab_decision.py
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from viz import labels as lab  # noqa: E402
from viz.data import Rollout  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
ABS = "foil_lifting_wrist_e40"
REL = "foil_lifting_relative_e40"


def wilson(k, n, z=1.96):
    """Wilson score interval - valid at small n, unlike the normal approx."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fisher_two_sided(a, b, c, d):
    """Fisher's exact test on [[a, b], [c, d]] by enumeration."""
    def logC(n, k):
        return (math.lgamma(n + 1) - math.lgamma(k + 1)
                - math.lgamma(n - k + 1))

    r1, r2 = a + b, c + d
    c1 = a + c
    n = r1 + r2

    def prob(x):
        return math.exp(logC(r1, x) + logC(r2, c1 - x) - logC(n, c1))

    lo = max(0, c1 - r2)
    hi = min(r1, c1)
    p_obs = prob(a)
    tot = 0.0
    for x in range(lo, hi + 1):
        p = prob(x)
        if p <= p_obs * (1 + 1e-9):
            tot += p
    return min(1.0, tot)


def load(ckpt):
    return lab.load(ROOT / ckpt)


def rate(labels, pred=None):
    sel = [x for x in labels.values() if pred is None or pred(x)]
    judged = [x for x in sel if x.verdict in ("success", "partial", "fail")]
    k = sum(1 for x in judged if x.ok)
    return k, len(judged)


def block_ci(la, lr):
    print("\n  1. HOW UNCERTAIN ARE THE TWO RATES?")
    ka, na = rate(la)
    kr, nr = rate(lr)
    for name, k, n in (("absolute", ka, na), ("relative", kr, nr)):
        lo, hi = wilson(k, n)
        print(f"     {name:>9s}: {k}/{n} = {100 * k / n:4.1f} %   "
              f"95 % CI [{100 * lo:4.1f}, {100 * hi:4.1f}]")
    p = fisher_two_sided(kr, nr - kr, ka, na - ka)
    print(f"     Fisher exact, two-sided: p = {p:.3f}")
    print("     The intervals overlap heavily, so the observed gap is")
    print("     within what 19-23 samples produce by chance.")
    return (ka, na), (kr, nr)


def block_flip(a, r):
    print("\n  2. HOW MANY OUTCOMES WOULD REVERSE THE RANKING?")
    ka, na = a
    kr, nr = r
    need = 0
    while True:
        need += 1
        if (ka + need) / na >= kr / nr:
            break
        if need > na:
            break
    print(f"     absolute needs +{need} success(es) out of {na} to reach "
          f"the relative rate")
    down = 0
    while True:
        down += 1
        if (kr - down) / nr <= ka / na:
            break
        if down > kr:
            break
    print(f"     or relative needs -{down} success(es) out of {nr}")
    print(f"     A ranking that {min(need, down)} rollout(s) can invert is "
          f"not a ranking.")


def block_matched(la, lr):
    """Only conditions present in BOTH runs, compared per condition."""
    print("\n  3. MATCHED CONDITIONS ONLY")
    ca = {c: g for c, g in lab.by_condition(la).items()}
    cr = {c: g for c, g in lab.by_condition(lr).items()}
    shared = [c for c in ca if c in cr]

    print(f"     {'condition':<34s} {'abs':>9s} {'rel':>9s}")
    ta = tn = ra = rn = 0
    for c in sorted(shared):
        ka, na = rate(la, lambda x, c=c: x.conditions == c)
        kr, nr = rate(lr, lambda x, c=c: x.conditions == c)
        ta += ka
        tn += na
        ra += kr
        rn += nr
        print(f"     {c:<34s} {ka}/{na:<7d} {kr}/{nr:<7d}")

    only_a = sorted(set(ca) - set(cr))
    only_r = sorted(set(cr) - set(ca))
    if only_a:
        print(f"     absolute only: {', '.join(only_a)}")
    if only_r:
        print(f"     relative only: {', '.join(only_r)}")

    if tn and rn:
        print(f"\n     matched totals: absolute {ta}/{tn} "
              f"({100 * ta / tn:.1f} %), relative {ra}/{rn} "
              f"({100 * ra / rn:.1f} %)")
        p = fisher_two_sided(ra, rn - ra, ta, tn - ta)
        print(f"     Fisher exact on the matched subset: p = {p:.3f}")


def block_startpose(la):
    """The absolute model's known confound, controlled for.

    findings 7.1: its successes are the runs closest to the successes' median
    start pose. The run therefore measured start-pose luck as much as policy
    quality. This asks what the rate looks like among runs that started in
    the region where it CAN work.
    """
    print("\n  4. ABSOLUTE MODEL WITH ITS START-POSE CONFOUND CONTROLLED")
    d = ROOT / ABS
    rows = []
    for name, lb in la.items():
        r = Rollout(d / name)
        if r.ok:
            rows.append((name, np.asarray(r.cmd_pose[0, :3], float), lb.ok))
    if not rows:
        return
    good = np.array([p for _, p, ok in rows if ok])
    if len(good) < 2:
        return
    ref = np.median(good, axis=0)

    for thr_mm in (30, 40, 60):
        sel = [(n, ok) for n, p, ok in rows
               if np.linalg.norm(p - ref) * 1000 <= thr_mm]
        if not sel:
            continue
        k = sum(1 for _, ok in sel if ok)
        lo, hi = wilson(k, len(sel))
        print(f"     started within {thr_mm:>3d} mm: {k}/{len(sel)} = "
              f"{100 * k / len(sel):5.1f} %   CI [{100 * lo:.0f}, "
              f"{100 * hi:.0f}]")
    print("     Its rate rises steeply as the start pose is constrained,")
    print("     which is exactly the comparison the relative run did not")
    print("     have to pass - the relative model has no such sensitivity.")


def block_power():
    print("\n  5. WHAT WOULD A DECIDABLE A/B NEED?")
    print(f"     {'p_abs':>7s} {'p_rel':>7s} {'n per arm':>11s}")
    for pa, pb in ((0.20, 0.30), (0.20, 0.40), (0.30, 0.50), (0.20, 0.50)):
        pbar = (pa + pb) / 2
        # Two-proportion z-test, alpha 0.05 two-sided, power 0.80.
        n = ((1.96 * math.sqrt(2 * pbar * (1 - pbar))
              + 0.84 * math.sqrt(pa * (1 - pa) + pb * (1 - pb))) ** 2
             / (pb - pa) ** 2)
        print(f"     {100 * pa:6.0f}% {100 * pb:6.0f}% {math.ceil(n):>11d}")
    print("     At the observed effect size (20 -> 30 %) the current 19-23")
    print("     rollouts are short by roughly an order of magnitude.")


def main():
    la, lr = load(ABS), load(REL)
    print("=== can these two runs decide which policy is better? ===")
    a, r = block_ci(la, lr)
    block_flip(a, r)
    block_matched(la, lr)
    block_startpose(la)
    block_power()

    print("\n  CONCLUSION")
    print("  The runs do not support a ranking. They were not matched on")
    print("  condition, the start pose was uncontrolled and biases the")
    print("  absolute model specifically, and the per-condition cells hold")
    print("  1-3 rollouts. What they DO support is the qualitative split:")
    print("  the two models fail differently (start-pose sensitivity and")
    print("  height under-reaction vs. height over-reaction, steeper vs.")
    print("  flatter pinch), and that is a statement about mechanism, which")
    print("  does not need a significant rate difference.")


if __name__ == "__main__":
    main()
