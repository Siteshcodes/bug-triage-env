# server/task.py
import sys
sys.path.insert(0, "/app")

from typing import Tuple, List
from model import BugReport, TriageAction
# ─────────────────────────────────────────────
# BUG REPORT DATASET
# ─────────────────────────────────────────────

TASKS = {
    "easy": {
        "bugs": [
            BugReport(
                id="easy-001",
                title="App crashes on login with correct credentials",
                body="When I enter my correct username and password, the app crashes immediately. "
                     "This started after the v2.0 release. Affects 100% of users.",
                author="user123",
                labels_hint=[],
                comments=["Confirmed on iOS and Android.", "Happens every time."],
            ),
            BugReport(
                id="easy-002",
                title="Typo in documentation homepage",
                body="There is a typo on the homepage docs: 'Welccome' should be 'Welcome'.",
                author="docs_fan",
                labels_hint=["documentation"],
                comments=[],
            ),
            BugReport(
                id="easy-003",
                title="Dashboard loads slowly for large datasets",
                body="When a dataset has more than 10k rows, the dashboard takes 30+ seconds to load.",
                author="power_user",
                labels_hint=["performance"],
                comments=["Noticed after the last deploy.", "CPU spikes to 100%."],
            ),
        ],
        # Ground truth for grader
        "answers": {
            "easy-001": {"priority": "P0"},
            "easy-002": {"priority": "P3"},
            "easy-003": {"priority": "P2"},
        },
    },

    "medium": {
        "bugs": [
            BugReport(
                id="med-001",
                title="Payment fails silently on checkout",
                body="Checkout completes without error but payment is never charged. "
                     "No error shown to user. Stripe logs show declined transaction.",
                author="store_owner",
                labels_hint=["bug"],
                comments=["Revenue impact confirmed.", "Happening since Tuesday."],
            ),
            BugReport(
                id="med-002",
                title="Search results include deleted posts",
                body="Deleted blog posts still appear in search results for up to 24 hours.",
                author="moderator_jane",
                labels_hint=[],
                comments=["GDPR concern — deleted content still visible."],
            ),
            BugReport(
                id="med-003",
                title="Dark mode toggle breaks layout on Safari",
                body="Switching to dark mode on Safari 16 causes nav bar to overlap content.",
                author="safari_user",
                labels_hint=["bug", "ux"],
                comments=["Only on Safari, not Chrome/Firefox."],
            ),
        ],
        "answers": {
            "med-001": {"priority": "P0", "labels": ["bug", "payments"]},
            "med-002": {"priority": "P1", "labels": ["bug", "security", "search"]},
            "med-003": {"priority": "P2", "labels": ["bug", "ux"]},
        },
    },

    "hard": {
        "bugs": [
            BugReport(
                id="hard-001",
                title="SQL injection vulnerability in search endpoint",
                body="The /api/search endpoint does not sanitize inputs. "
                     "Crafted queries can dump user table. PoC attached.",
                author="security_researcher",
                labels_hint=[],
                comments=["Critical. Affects production.", "Do not discuss publicly."],
            ),
            BugReport(
                id="hard-002",
                title="Memory leak in background job processor causes OOM after 6 hours",
                body="The job processor allocates ~50MB per job and never frees it. "
                     "Server runs out of memory every 6 hours, requiring restart.",
                author="devops_alice",
                labels_hint=["performance"],
                comments=["Verified with heap profiler.", "Started in v1.9."],
            ),
            BugReport(
                id="hard-003",
                title="Race condition in file upload: files occasionally overwrite each other",
                body="Under concurrent load, two users uploading simultaneously sometimes "
                     "get each other's files. Rare (1 in 10k uploads) but data integrity risk.",
                author="qa_bot",
                labels_hint=["bug"],
                comments=["Reproduced with locust at 50 concurrent users."],
            ),
        ],
        "answers": {
            "hard-001": {
                "priority": "P0",
                "labels": ["bug", "security"],
                "assigned_team": "security",
                "milestone": "hotfix",
            },
            "hard-002": {
                "priority": "P1",
                "labels": ["bug", "performance"],
                "assigned_team": "backend",
                "milestone": "v2.1",
            },
            "hard-003": {
                "priority": "P1",
                "labels": ["bug", "data-integrity"],
                "assigned_team": "backend",
                "milestone": "v2.1",
            },
        },
    },
}


# ─────────────────────────────────────────────
# GRADERS
# ─────────────────────────────────────────────

PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}

def _priority_score(predicted: str, correct: str) -> float:
    """Exact match = 1.0, one level off = 0.5, two+ off = 0.0"""
    if predicted == correct:
        return 1.0
    diff = abs(PRIORITY_ORDER.get(predicted, 99) - PRIORITY_ORDER.get(correct, 99))
    return 0.5 if diff == 1 else 0.0


def _label_score(predicted: List[str], correct: List[str]) -> float:
    """Jaccard similarity between predicted and correct label sets."""
    pred_set = set(l.lower() for l in predicted)
    corr_set = set(l.lower() for l in correct)
    if not corr_set:
        return 1.0
    intersection = pred_set & corr_set
    union = pred_set | corr_set
    return len(intersection) / len(union)


def grade_action(
    task_key: str, bug: BugReport, action: TriageAction
) -> Tuple[float, str]:
    """
    Returns (score: 0.0–1.0, feedback: str)
    """
    answer = TASKS[task_key]["answers"][bug.id]
    feedback_parts = []

    if task_key == "easy":
        # Only grade priority
        score = _priority_score(action.priority, answer["priority"])
        feedback_parts.append(
            f"Priority: {'✓' if score == 1.0 else '~' if score == 0.5 else '✗'} "
            f"(got {action.priority}, expected {answer['priority']})"
        )
        return score, " | ".join(feedback_parts)

    elif task_key == "medium":
        # Grade priority (50%) + labels (50%)
        p_score = _priority_score(action.priority, answer["priority"])
        l_score = _label_score(action.labels, answer["labels"])
        score = 0.5 * p_score + 0.5 * l_score
        feedback_parts.append(f"Priority score: {p_score:.2f}")
        feedback_parts.append(f"Label score: {l_score:.2f}")
        return round(score, 3), " | ".join(feedback_parts)

    else:  # hard
        # Grade priority (40%) + labels (35%) + team (25%)
        p_score = _priority_score(action.priority, answer["priority"])
        l_score = _label_score(action.labels, answer["labels"])
        t_score = 1.0 if action.assigned_team.lower() == answer["assigned_team"].lower() else 0.0
        score = 0.40 * p_score + 0.35 * l_score + 0.25 * t_score
        feedback_parts.append(f"Priority: {p_score:.2f}")
        feedback_parts.append(f"Labels: {l_score:.2f}")
        feedback_parts.append(f"Team: {t_score:.2f}")
        # Bonus: penalize missing security escalation
        if answer.get("assigned_team") == "security" and action.assigned_team.lower() != "security":
            score = max(0.0, score - 0.15)
            feedback_parts.append("⚠ Security escalation missed (-0.15)")
        return round(score, 3), " | ".join(feedback_parts)