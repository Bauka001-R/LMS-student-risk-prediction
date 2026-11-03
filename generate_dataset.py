"""Generate a SYNTHETIC student-activity dataset.

The real LMS data used in the thesis contains personal information of students and
is not published. This script produces a dataset with the same structure (22 features,
247 students, 18 groups, ~29% at-risk) so that the whole pipeline can be reproduced.

Two latent variables drive everything: *engagement* (how actively a student works in
the LMS) and *ability* (academic level). Features are noisy functions of these two,
and the at-risk label is a noisy function of them as well.

Usage:
    python generate_dataset.py                 # writes data/students.csv
    python generate_dataset.py --n 500 --seed 1
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from features import FEATURE_NAMES, SEED, TARGET


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(n_students: int = 247, n_groups: int = 18, risk_share: float = 0.29,
             seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    group = rng.integers(0, n_groups, n_students)
    group_effect = rng.normal(0, 0.3, n_groups)

    eng = rng.normal(0, 1, n_students) + group_effect[group]          # engagement
    abil = 0.5 * eng + np.sqrt(0.75) * rng.normal(0, 1, n_students)    # ability

    def noise(scale):
        return rng.normal(0, scale, n_students)

    d = {}
    # --- activity / clickstream ---
    d["login_count"] = rng.poisson(np.exp(4.0 + 0.35 * eng))
    d["session_duration_mean"] = np.clip(22 + 5 * eng + noise(5), 3, 60)
    d["session_duration_std"] = np.clip(9 - 1.5 * eng + noise(3), 1, 30)
    d["days_since_last_activity"] = np.clip(np.round(rng.exponential(np.exp(1.0 - 0.7 * eng))), 0, 60)
    d["active_share_last_7d"] = np.clip(_sigmoid(-1.0 + 0.8 * eng + noise(0.6)), 0, 1)
    d["activity_trend"] = 0.3 * eng + noise(0.5)
    d["night_activity_ratio"] = np.clip(0.12 - 0.02 * eng + noise(0.05), 0, 0.5)
    d["reading_time_ratio"] = np.clip(0.9 * _sigmoid(0.9 * eng + noise(0.6)), 0, 1)
    d["material_revisit_rate"] = np.clip(0.25 + 0.08 * eng + noise(0.1), 0, 1)
    d["scroll_max_depth"] = np.clip(0.6 + 0.12 * eng + noise(0.15), 0.05, 1)
    d["hesitation_score"] = np.clip(0.5 - 0.1 * eng + noise(0.15), 0, 1)
    # --- assignments / quizzes ---
    d["assignment_mean_score"] = np.clip(65 + 10 * abil + 4 * eng + noise(8), 0, 100)
    d["assignment_score_std"] = np.clip(12 - 2 * abil + noise(3), 1, 30)
    d["assignment_ontime_ratio"] = np.clip(_sigmoid(0.4 + 0.9 * eng + 0.4 * abil + noise(0.6)), 0, 1)
    d["assignment_lastminute_ratio"] = np.clip(0.3 - 0.1 * eng + noise(0.12), 0, 1)
    d["quiz_mean_score"] = np.clip(60 + 11 * abil + 3 * eng + noise(10), 0, 100)
    d["quiz_time_mean"] = np.clip(14 - 1.5 * abil + noise(3), 3, 40)
    d["quiz_retakes"] = rng.poisson(np.exp(0.3 - 0.3 * abil))
    # --- profile / attendance ---
    d["baseline_gpa"] = np.clip(2.9 + 0.45 * abil + noise(0.45), 1.0, 4.0)
    attendance = np.clip(_sigmoid(1.2 + 1.0 * eng + noise(0.6)), 0, 1)
    d["attendance_ratio"] = attendance
    d["attendance_last2w"] = np.clip(attendance + 0.03 * eng + noise(0.1), 0, 1)
    d["max_absent_streak"] = rng.poisson(np.exp(0.7 - 0.6 * eng))

    # --- label: noisy function of the latent variables; top `risk_share` are at risk ---
    latent_risk = -(0.9 * abil + 0.8 * eng) + noise(0.6)
    cutoff = np.quantile(latent_risk, 1 - risk_share)
    at_risk = (latent_risk >= cutoff).astype(int)

    df = pd.DataFrame(d)[FEATURE_NAMES].round(4)
    df.insert(0, "group", [f"GRP-{g + 1:02d}" for g in group])
    df.insert(0, "student_id", [f"STU-{i + 1:04d}" for i in range(n_students)])
    df[TARGET] = at_risk
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=247, help="number of students")
    p.add_argument("--groups", type=int, default=18)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", default="data/students.csv")
    a = p.parse_args()

    df = generate(a.n, a.groups, seed=a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} students ({df[TARGET].mean():.1%} at risk) -> {out}")


if __name__ == "__main__":
    main()
