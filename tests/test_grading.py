# tests/test_grading.py
"""Tests for the grading logic in server/task.py"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "server"))

import pytest
from model import BugReport, TriageAction
from server.task import (
    _priority_score, _label_score, _normalize_label, _reasoning_score,
    grade_action, generate_bug, sample_bug, TASKS, LABEL_SYNONYMS,
)


# ── Priority Scoring ──────────────────────────────────────

class TestPriorityScoring:
    def test_exact_match_gives_high_score(self):
        assert _priority_score("P0", "P0") == 0.95

    def test_all_exact_matches(self):
        for p in ["P0", "P1", "P2", "P3"]:
            assert _priority_score(p, p) == 0.95

    def test_off_by_one_gives_partial_credit(self):
        assert _priority_score("P0", "P1") == 0.5
        assert _priority_score("P1", "P2") == 0.5
        assert _priority_score("P2", "P3") == 0.5

    def test_off_by_two_gives_low_credit(self):
        assert _priority_score("P0", "P2") == 0.2
        assert _priority_score("P1", "P3") == 0.2

    def test_completely_wrong_gives_minimum(self):
        assert _priority_score("P0", "P3") == 0.05

    def test_invalid_priority(self):
        assert _priority_score("P9", "P0") == 0.05
        assert _priority_score("invalid", "P0") == 0.05


# ── Label Scoring ─────────────────────────────────────────

class TestLabelScoring:
    def test_perfect_match(self):
        score = _label_score(["bug", "security"], ["bug", "security"])
        assert score >= 0.9

    def test_partial_overlap(self):
        score = _label_score(["bug"], ["bug", "security"])
        assert 0.3 < score < 0.7  # ~50% Jaccard

    def test_no_overlap(self):
        score = _label_score(["docs"], ["bug", "security"])
        assert score == 0.05  # clamped minimum

    def test_empty_correct_labels(self):
        score = _label_score(["bug"], [])
        assert score == 0.95  # nothing expected => full credit

    def test_synonym_matching(self):
        # "defect" is a synonym for "bug"
        score = _label_score(["defect"], ["bug"])
        assert score >= 0.9  # should match via synonym

    def test_case_insensitive(self):
        score = _label_score(["BUG", "Security"], ["bug", "security"])
        assert score >= 0.9


# ── Label Normalization ───────────────────────────────────

class TestLabelNormalization:
    def test_canonical_stays_same(self):
        assert _normalize_label("bug") == "bug"
        assert _normalize_label("security") == "security"

    def test_synonym_maps_to_canonical(self):
        assert _normalize_label("defect") == "bug"
        assert _normalize_label("vulnerability") == "security"
        assert _normalize_label("slow") == "performance"
        assert _normalize_label("ui") == "ux"

    def test_unknown_label_passes_through(self):
        assert _normalize_label("my-custom-label") == "my-custom-label"

    def test_case_insensitive(self):
        assert _normalize_label("BUG") == "bug"
        assert _normalize_label("Vulnerability") == "security"


# ── Reasoning Scoring ─────────────────────────────────────

class TestReasoningScoring:
    def test_empty_reasoning_gives_zero(self):
        assert _reasoning_score("", {"priority": "P0"}) == 0.0

    def test_short_reasoning_gives_zero(self):
        assert _reasoning_score("bad", {"priority": "P0"}) == 0.0

    def test_relevant_reasoning_gives_bonus(self):
        score = _reasoning_score(
            "This is a critical security vulnerability affecting production and causing data loss",
            {"priority": "P0"},
        )
        assert score > 0

    def test_bonus_capped_at_max(self):
        score = _reasoning_score(
            "production down all users data loss security crash revenue injection vulnerability 100%",
            {"priority": "P0"},
        )
        assert score <= 0.15


# ── Grade Action ──────────────────────────────────────────

class TestGradeAction:
    @pytest.fixture
    def easy_bug(self):
        return TASKS["easy"]["bugs"][0]  # easy-001: P0

    @pytest.fixture
    def medium_bug(self):
        return TASKS["medium"]["bugs"][0]  # med-001: P0, payments, backend

    @pytest.fixture
    def hard_bug(self):
        return TASKS["hard"]["bugs"][0]  # hard-001: P0, security, hotfix

    def test_easy_perfect_answer(self, easy_bug):
        action = TriageAction(priority="P0")
        score, feedback = grade_action("easy", easy_bug, action)
        assert 0.9 <= score <= 0.99
        assert "✓" in feedback

    def test_easy_wrong_answer(self, easy_bug):
        action = TriageAction(priority="P3")
        score, feedback = grade_action("easy", easy_bug, action)
        assert score < 0.2

    def test_medium_perfect_answer(self, medium_bug):
        action = TriageAction(
            priority="P0",
            labels=["bug", "payments"],
            assigned_team="backend",
        )
        score, feedback = grade_action("medium", medium_bug, action)
        assert score > 0.8

    def test_hard_security_penalty(self, hard_bug):
        # hard-001 requires security team; assigning backend should be penalized
        action_wrong = TriageAction(
            priority="P0",
            labels=["bug", "security"],
            assigned_team="backend",  # Wrong! Should be security
            milestone="hotfix",
        )
        action_right = TriageAction(
            priority="P0",
            labels=["bug", "security"],
            assigned_team="security",
            milestone="hotfix",
        )
        score_wrong, fb_wrong = grade_action("hard", hard_bug, action_wrong)
        score_right, fb_right = grade_action("hard", hard_bug, action_right)

        assert score_right > score_wrong
        assert "Security escalation missed" in fb_wrong

    def test_all_scores_in_valid_range(self):
        """Every grading result must be in (0, 1) — open interval."""
        for task_key in ["easy", "medium", "hard"]:
            for bug in TASKS[task_key]["bugs"]:
                for priority in ["P0", "P1", "P2", "P3"]:
                    action = TriageAction(
                        priority=priority,
                        labels=["bug"],
                        assigned_team="backend",
                        milestone="backlog",
                    )
                    score, feedback = grade_action(task_key, bug, action)
                    assert 0 < score < 1, (
                        f"Score {score} out of range for {bug.id} "
                        f"with priority={priority}"
                    )
                    assert isinstance(feedback, str)
                    assert len(feedback) > 0


# ── Procedural Bug Generation ─────────────────────────────

class TestBugGeneration:
    def test_generate_produces_valid_bug(self):
        bug, answer = generate_bug("easy", seed=42)
        assert isinstance(bug, BugReport)
        assert bug.id.startswith("gen-")
        assert len(bug.title) > 5
        assert len(bug.body) > 20
        assert "priority" in answer

    def test_different_seeds_produce_different_bugs(self):
        bug1, _ = generate_bug("easy", seed=1)
        bug2, _ = generate_bug("easy", seed=2)
        # Very unlikely to produce the same title with different seeds
        assert bug1.title != bug2.title or bug1.body != bug2.body

    def test_same_seed_produces_same_bug(self):
        bug1, ans1 = generate_bug("easy", seed=42)
        bug2, ans2 = generate_bug("easy", seed=42)
        assert bug1.title == bug2.title
        assert bug1.body == bug2.body
        assert ans1 == ans2

    def test_easy_bugs_have_only_priority(self):
        for seed in range(10):
            _, answer = generate_bug("easy", seed=seed)
            assert "priority" in answer
            # easy should NOT include milestone
            assert "milestone" not in answer

    def test_hard_bugs_have_full_answer(self):
        for seed in range(50):
            _, answer = generate_bug("hard", seed=seed)
            assert "priority" in answer

    def test_all_difficulties(self):
        for difficulty in ["easy", "medium", "hard"]:
            bug, answer = generate_bug(difficulty, seed=100)
            assert isinstance(bug, BugReport)
            assert "priority" in answer

    def test_sample_bug_returns_tuple(self):
        bug, answer = sample_bug("easy", seed=42)
        assert isinstance(bug, BugReport)
        assert isinstance(answer, dict)

    def test_generated_bugs_are_gradeable(self):
        """Generated bugs should work with the grading system."""
        for difficulty in ["easy", "medium", "hard"]:
            for seed in range(5):
                bug, answer = generate_bug(difficulty, seed=seed)
                action = TriageAction(
                    priority=answer["priority"],
                    labels=answer.get("labels", ["bug"]),
                    assigned_team=answer.get("assigned_team", "backend"),
                    milestone=answer.get("milestone", "backlog"),
                )
                score, feedback = grade_action(difficulty, bug, action, answer=answer)
                assert 0 < score < 1, (
                    f"Score {score} for {bug.id} ({difficulty})"
                )
