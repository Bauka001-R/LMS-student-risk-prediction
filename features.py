"""Shared definitions: the 22 engineered features and risk-level thresholds."""

SEED = 42  # same random_state as used throughout the thesis experiments

# (name, description) - the hybrid feature set combining clickstream and structured data
FEATURES = [
    # --- activity / clickstream ---
    ("login_count", "Total number of logins during the observation window"),
    ("session_duration_mean", "Mean session duration, minutes"),
    ("session_duration_std", "Std. deviation of session duration (instability of work pattern)"),
    ("days_since_last_activity", "Days since the last recorded activity"),
    ("active_share_last_7d", "Share of activity that happened in the last 7 days"),
    ("activity_trend", "Second-half vs first-half activity difference (normalised)"),
    ("night_activity_ratio", "Share of events between 00:00 and 06:00"),
    ("reading_time_ratio", "Time spent reading materials / time of all events"),
    ("material_revisit_rate", "Share of materials that were opened more than once"),
    ("scroll_max_depth", "Mean maximum scroll depth in learning materials (0-1)"),
    ("hesitation_score", "Indicator built from pauses between consecutive actions (0-1)"),
    # --- assignments / quizzes (structured data) ---
    ("assignment_mean_score", "Mean assignment score, 0-100"),
    ("assignment_score_std", "Std. deviation of assignment scores"),
    ("assignment_ontime_ratio", "Share of assignments submitted on time"),
    ("assignment_lastminute_ratio", "Share of assignments submitted in the last hours before the deadline"),
    ("quiz_mean_score", "Mean first-attempt quiz score, 0-100"),
    ("quiz_time_mean", "Mean time spent on a quiz, minutes"),
    ("quiz_retakes", "Number of quiz retakes"),
    # --- profile / attendance ---
    ("baseline_gpa", "Baseline academic performance (GPA, 4.0 scale)"),
    ("attendance_ratio", "Overall attendance share"),
    ("attendance_last2w", "Attendance share over the last two weeks"),
    ("max_absent_streak", "Longest streak of missed classes in a row"),
]

FEATURE_NAMES = [name for name, _ in FEATURES]
assert len(FEATURE_NAMES) == 22

TARGET = "at_risk"

# Risk-level bands used by the API (and by the dashboard traffic lights)
LOW_MAX = 0.33
MEDIUM_MAX = 0.66


def risk_level(probability: float) -> str:
    """Map a probability to the three bands shown to the teacher."""
    if probability < LOW_MAX:
        return "low"
    if probability < MEDIUM_MAX:
        return "medium"
    return "high"
