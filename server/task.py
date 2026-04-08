# server/task.py
import sys
import random
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
                     "This started after the v2.0 release. Affects 100% of users. "
                     "No workaround exists — users cannot log in at all.",
                author="user123",
                labels_hint=[],
                comments=["Confirmed on iOS and Android.", "Happens every time."],
            ),
            BugReport(
                id="easy-002",
                title="Typo in documentation homepage",
                body="There is a typo on the homepage docs: 'Welccome' should be 'Welcome'. "
                     "No functional impact, purely cosmetic.",
                author="docs_fan",
                labels_hint=["documentation"],
                comments=[],
            ),
            BugReport(
                id="easy-003",
                title="Dashboard loads slowly for large datasets",
                body="When a dataset has more than 10k rows, the dashboard takes 30+ seconds to load. "
                     "Workaround: export data and use offline tools. Affects power users only.",
                author="power_user",
                labels_hint=["performance"],
                comments=["Noticed after the last deploy.", "CPU spikes to 100%."],
            ),
            BugReport(
                id="easy-004",
                title="Email notifications not sent after password reset",
                body="Users who reset their password do not receive the confirmation email. "
                     "SMTP logs show the job is queued but never dispatched. "
                     "Affects all users attempting password reset.",
                author="support_team",
                labels_hint=["bug"],
                comments=["Reported by 12 users this week.", "Started after email service migration."],
            ),
            BugReport(
                id="easy-005",
                title="Incorrect copyright year in footer",
                body="The footer shows '© 2022' but it should be '© 2024'. "
                     "No functional impact.",
                author="intern_dev",
                labels_hint=["documentation"],
                comments=[],
            ),
        ],
        # Ground truth for grader
        "answers": {
            "easy-001": {"priority": "P0"},
            "easy-002": {"priority": "P3"},
            "easy-003": {"priority": "P2"},
            "easy-004": {"priority": "P1"},
            "easy-005": {"priority": "P3"},
        },
    },

    "medium": {
        "bugs": [
            BugReport(
                id="med-001",
                title="Payment fails silently on checkout",
                body="Checkout completes without error but payment is never charged. "
                     "No error shown to user. Stripe logs show declined transaction. "
                     "Direct revenue loss — every failed checkout is a lost sale.",
                author="store_owner",
                labels_hint=["bug"],
                comments=["Revenue impact confirmed.", "Happening since Tuesday."],
            ),
            BugReport(
                id="med-002",
                title="Search results include deleted posts",
                body="Deleted blog posts still appear in search results for up to 24 hours. "
                     "Users can read content that was explicitly removed by moderators. "
                     "Potential GDPR violation if deleted content belongs to EU users.",
                author="moderator_jane",
                labels_hint=[],
                comments=["GDPR concern — deleted content still visible."],
            ),
            BugReport(
                id="med-003",
                title="Dark mode toggle breaks layout on Safari",
                body="Switching to dark mode on Safari 16 causes nav bar to overlap content. "
                     "Chrome and Firefox unaffected. Workaround: use a different browser.",
                author="safari_user",
                labels_hint=["bug", "ux"],
                comments=["Only on Safari, not Chrome/Firefox."],
            ),
            BugReport(
                id="med-004",
                title="CSV export produces corrupted file for non-ASCII characters",
                body="When table data contains accented characters (e.g. café, naïve), "
                     "the exported CSV file is corrupted and cannot be opened in Excel. "
                     "Affects users with international data.",
                author="data_analyst",
                labels_hint=["bug"],
                comments=["Encoding issue — UTF-8 not respected.", "Workaround: manual copy-paste."],
            ),
            BugReport(
                id="med-005",
                title="API rate limiter blocks legitimate users after 429 error",
                body="After receiving a 429 Too Many Requests response, legitimate users "
                     "remain blocked for 1 hour even after the rate limit window resets. "
                     "The unblock logic has a bug — it never clears the blocked flag.",
                author="api_user",
                labels_hint=["bug"],
                comments=["Affects CI/CD pipelines hitting the API.", "Retry-After header is wrong."],
            ),
        ],
        "answers": {
            "med-001": {"priority": "P0", "labels": ["bug", "payments"],        "assigned_team": "backend"},
            "med-002": {"priority": "P1", "labels": ["bug", "security"],        "assigned_team": "security"},
            "med-003": {"priority": "P2", "labels": ["bug", "ux"],              "assigned_team": "frontend"},
            "med-004": {"priority": "P2", "labels": ["bug", "data-integrity"],  "assigned_team": "backend"},
            "med-005": {"priority": "P1", "labels": ["bug", "performance"],     "assigned_team": "backend"},
        },
    },

    "hard": {
        "bugs": [
            BugReport(
                id="hard-001",
                title="SQL injection vulnerability in search endpoint",
                body="The /api/search endpoint does not sanitize inputs. "
                     "Crafted queries can dump the entire user table including password hashes. "
                     "PoC attached. Verified on production. Treat as confidential — "
                     "do not discuss publicly until patched.",
                author="security_researcher",
                labels_hint=[],
                comments=["Critical. Affects production.", "Do not discuss publicly."],
            ),
            BugReport(
                id="hard-002",
                title="Memory leak in background job processor causes OOM after 6 hours",
                body="The job processor allocates ~50MB per job and never frees it. "
                     "Server runs out of memory every 6 hours, requiring a manual restart. "
                     "Heap profiler confirms leak introduced in v1.9. "
                     "Workaround: scheduled restarts every 4 hours (operational overhead).",
                author="devops_alice",
                labels_hint=["performance"],
                comments=["Verified with heap profiler.", "Started in v1.9."],
            ),
            BugReport(
                id="hard-003",
                title="Race condition in file upload: files occasionally overwrite each other",
                body="Under concurrent load, two users uploading simultaneously can get "
                     "each other's files due to a race condition in the temp file naming logic. "
                     "Frequency: approximately 1 in 10,000 uploads under normal load. "
                     "No data loss confirmed yet and a workaround exists: "
                     "enable sequential upload mode in settings (disabled by default). "
                     "Risk is low-probability but affects data integrity.",
                author="qa_bot",
                labels_hint=["bug"],
                comments=["Reproduced with locust at 50 concurrent users.", "Sequential mode avoids it."],
            ),
            BugReport(
                id="hard-004",
                title="Auth token not invalidated after password change",
                body="When a user changes their password, existing JWT tokens remain valid "
                     "for up to 24 hours. An attacker who previously stole a token can "
                     "continue to access the account even after the password is reset. "
                     "This is a session management security vulnerability.",
                author="pentest_team",
                labels_hint=["security"],
                comments=["Verified on staging.", "OWASP A07 — Identification and Authentication Failures."],
            ),
            BugReport(
                id="hard-005",
                title="Infinite loop in webhook retry logic causes CPU spike",
                body="When a webhook endpoint returns a 500 error, the retry logic enters "
                     "an infinite loop with no backoff or retry cap. "
                     "This causes CPU to spike to 100% within minutes and starves other services. "
                     "Triggered in production twice this week. Requires process kill to recover.",
                author="oncall_eng",
                labels_hint=["bug", "performance"],
                comments=["PagerDuty alert fired twice.", "Needs exponential backoff + max retry cap."],
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
            "hard-004": {
                "priority": "P0",
                "labels": ["bug", "security"],
                "assigned_team": "security",
                "milestone": "hotfix",
            },
            "hard-005": {
                "priority": "P0",
                "labels": ["bug", "performance"],
                "assigned_team": "backend",
                "milestone": "hotfix",
            },
        },
    },
}


# ─────────────────────────────────────────────
# TASK SAMPLER  — picks a random bug each reset
# ─────────────────────────────────────────────

def sample_bug(task_key: str) -> BugReport:
    """Return a random bug from the given task's pool."""
    return random.choice(TASKS[task_key]["bugs"])


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

    Easy   — priority only (100%)
    Medium — priority (45%) + labels (40%) + team routing (15%)
    Hard   — priority (35%) + labels (30%) + team (20%) + milestone (15%)
             with -0.15 penalty for missing security escalation
    """
    answer = TASKS[task_key]["answers"][bug.id]
    feedback_parts = []

    if task_key == "easy":
        # Only grade priority
        score = _priority_score(action.priority, answer["priority"])
        symbol = "✓" if score == 1.0 else "~" if score == 0.5 else "✗"
        feedback_parts.append(
            f"Priority: {symbol} (got {action.priority}, expected {answer['priority']})"
        )
        return round(score, 3), " | ".join(feedback_parts)

    elif task_key == "medium":
        # Priority (45%) + labels (40%) + team routing (15%)
        p_score = _priority_score(action.priority, answer["priority"])
        l_score = _label_score(action.labels, answer["labels"])

        expected_team = answer.get("assigned_team", "")
        t_score = (
            1.0
            if expected_team and action.assigned_team.lower() == expected_team.lower()
            else 0.0
        )

        score = 0.45 * p_score + 0.40 * l_score + 0.15 * t_score

        feedback_parts.append(f"Priority: {p_score:.2f} (got {action.priority}, expected {answer['priority']})")
        feedback_parts.append(f"Labels: {l_score:.2f}")
        feedback_parts.append(f"Team: {t_score:.2f} (got {action.assigned_team}, expected {expected_team})")

        return round(score, 3), " | ".join(feedback_parts)

    else:  # hard
        # Priority (35%) + labels (30%) + team (20%) + milestone (15%)
        p_score = _priority_score(action.priority, answer["priority"])
        l_score = _label_score(action.labels, answer["labels"])
        t_score = (
            1.0
            if action.assigned_team.lower() == answer["assigned_team"].lower()
            else 0.0
        )
        m_score = (
            1.0
            if action.milestone.lower() == answer["milestone"].lower()
            else 0.0
        )

        score = 0.35 * p_score + 0.30 * l_score + 0.20 * t_score + 0.15 * m_score

        feedback_parts.append(f"Priority: {p_score:.2f} (got {action.priority}, expected {answer['priority']})")
        feedback_parts.append(f"Labels: {l_score:.2f}")
        feedback_parts.append(f"Team: {t_score:.2f} (got {action.assigned_team}, expected {answer['assigned_team']})")
        feedback_parts.append(f"Milestone: {m_score:.2f} (got {action.milestone}, expected {answer['milestone']})")

        # Penalty: missing security escalation on security bugs
        if answer.get("assigned_team") == "security" and action.assigned_team.lower() != "security":
            score = max(0.0, score - 0.15)
            feedback_parts.append("⚠ Security escalation missed (-0.15)")

        return round(score, 3), " | ".join(feedback_parts)
    
def priority_match(*args, **kwargs):
    if len(args) < 2:
        return 0.0

    bug = args[0]
    action = args[1]

    score, _ = grade_action("easy", bug, action)
    return float(score)


def priority_label_team(*args, **kwargs):
    if len(args) < 2:
        return 0.0

    bug = args[0]
    action = args[1]

    score, _ = grade_action("medium", bug, action)
    return float(score)


def full_triage(*args, **kwargs):
    if len(args) < 2:
        return 0.0

    bug = args[0]
    action = args[1]

    score, _ = grade_action("hard", bug, action)
    return float(score)
__all__ = [
    "priority_match",
    "priority_label_team",
    "full_triage",
    "sample_bug",
    "grade_action",
    "TASKS", 
]