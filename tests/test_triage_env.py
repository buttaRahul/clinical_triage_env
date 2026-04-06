"""
Tests for Clinical Triage Navigator.
Run with: pytest tests/ -v
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import TriageAction, TriageObservation, TriageState
from server.clinical_triage_environment import ClinicalTriageEnvironment


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def easy_env():
    return ClinicalTriageEnvironment(task="easy")

@pytest.fixture
def medium_env():
    return ClinicalTriageEnvironment(task="medium")

@pytest.fixture
def hard_env():
    return ClinicalTriageEnvironment(task="hard")


def perfect_action(env: ClinicalTriageEnvironment) -> TriageAction:
    """Return a perfect action for the current patient."""
    case = env._cases[env._queue_index]
    return TriageAction(
        esi_level=case["ground_truth_esi"],
        resources_requested=case["ground_truth_resources"],
        disposition=case["ground_truth_disposition"],
        clinical_reasoning="Perfect ground truth action.",
    )


def wrong_action(gt_esi: int) -> TriageAction:
    """Return an action with ESI off by 2 (under-triage)."""
    bad_esi = min(5, gt_esi + 2)
    return TriageAction(
        esi_level=bad_esi,
        resources_requested=["LABS"],
        disposition="waiting_room",
        clinical_reasoning="Intentionally wrong for testing.",
    )


# ──────────────────────────────────────────────────────────────
# reset() tests
# ──────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_returns_observation(self, easy_env):
        obs = easy_env.reset()
        assert isinstance(obs, TriageObservation)

    def test_reset_has_first_patient(self, easy_env):
        obs = easy_env.reset()
        assert obs.current_patient is not None
        assert obs.current_patient.patient_id == "E001"

    def test_reset_not_done(self, easy_env):
        obs = easy_env.reset()
        assert obs.done is False

    def test_reset_zero_reward(self, easy_env):
        obs = easy_env.reset()
        assert obs.cumulative_reward == 0.0
        assert obs.step_reward == 0.0

    def test_reset_clears_previous_episode(self, easy_env):
        easy_env.reset()
        action = perfect_action(easy_env)
        easy_env.step(action)
        # Reset again — should start fresh
        obs = easy_env.reset()
        assert obs.patients_triaged == 0
        assert obs.cumulative_reward == 0.0

    def test_reset_10_patients_remaining_easy(self, easy_env):
        obs = easy_env.reset()
        assert obs.patients_remaining == 10

    def test_reset_task_description_populated(self, easy_env):
        obs = easy_env.reset()
        assert len(obs.task_description) > 0
        assert obs.task_name == "easy"


# ──────────────────────────────────────────────────────────────
# step() tests
# ──────────────────────────────────────────────────────────────

class TestStep:
    def test_step_requires_reset_first(self):
        env = ClinicalTriageEnvironment(task="easy")
        with pytest.raises(RuntimeError):
            env.step(TriageAction(esi_level=3, disposition="monitored_bed"))

    def test_perfect_action_positive_reward(self, easy_env):
        easy_env.reset()
        action = perfect_action(easy_env)
        obs = easy_env.step(action)
        assert obs.step_reward > 0.0

    def test_exact_esi_match_reward_above_1(self, easy_env):
        easy_env.reset()
        action = perfect_action(easy_env)
        obs = easy_env.step(action)
        # Exact ESI + resources + disposition > 1.0
        assert obs.step_reward >= 1.0

    def test_wrong_esi_negative_reward(self, easy_env):
        easy_env.reset()
        gt_esi = easy_env._cases[0]["ground_truth_esi"]
        obs = easy_env.step(wrong_action(gt_esi))
        assert obs.step_reward < 0.0

    def test_under_triage_penalized_more(self, easy_env):
        """Under-triage (too high ESI number) should cost more than over-triage."""
        easy_env.reset()
        gt_esi = easy_env._cases[0]["ground_truth_esi"]

        # Under-triage: assign higher ESI number (less urgent)
        if gt_esi < 4:
            easy_env.reset()
            under_action = TriageAction(
                esi_level=gt_esi + 2, disposition="waiting_room",
                resources_requested=["LABS"]
            )
            obs_under = easy_env.step(under_action)
            under_reward = obs_under.step_reward

            easy_env.reset()
            over_action = TriageAction(
                esi_level=max(1, gt_esi - 2), disposition="resus_bay",
                resources_requested=["IV_ACCESS", "LABS", "ECG"]
            )
            obs_over = easy_env.step(over_action)
            over_reward = obs_over.step_reward

            assert under_reward < over_reward  # under-triage punished harder

    def test_step_advances_queue(self, easy_env):
        easy_env.reset()
        action = perfect_action(easy_env)
        obs = easy_env.step(action)
        assert obs.patients_triaged == 1
        assert obs.patients_remaining == 9

    def test_step_feedback_populated(self, easy_env):
        easy_env.reset()
        action = perfect_action(easy_env)
        obs = easy_env.step(action)
        assert len(obs.last_action_feedback) > 0

    def test_step_esi_correct_flag(self, easy_env):
        easy_env.reset()
        action = perfect_action(easy_env)
        obs = easy_env.step(action)
        assert obs.last_esi_correct is True

    def test_full_easy_episode_ends_done(self, easy_env):
        easy_env.reset()
        for _ in range(10):
            action = perfect_action(easy_env)
            obs = easy_env.step(action)
        assert obs.done is True
        assert obs.patients_remaining == 0

    def test_step_after_done_raises(self, easy_env):
        easy_env.reset()
        for _ in range(10):
            obs = easy_env.step(perfect_action(easy_env))
        with pytest.raises(RuntimeError, match="done"):
            easy_env.step(TriageAction(esi_level=3, disposition="monitored_bed"))


# ──────────────────────────────────────────────────────────────
# state() tests
# ──────────────────────────────────────────────────────────────

class TestState:
    def test_state_returns_triage_state(self, easy_env):
        easy_env.reset()
        s = easy_env.state
        assert isinstance(s, TriageState)

    def test_state_episode_id_set_after_reset(self, easy_env):
        easy_env.reset()
        s = easy_env.state
        assert len(s.episode_id) > 0

    def test_state_counts_update(self, easy_env):
        easy_env.reset()
        assert easy_env.state.patients_triaged == 0
        easy_env.step(perfect_action(easy_env))
        assert easy_env.state.patients_triaged == 1
        easy_env.step(perfect_action(easy_env))
        assert easy_env.state.patients_triaged == 2

    def test_state_accuracy_perfect(self, easy_env):
        easy_env.reset()
        for _ in range(5):
            easy_env.step(perfect_action(easy_env))
        s = easy_env.state
        assert s.esi_accuracy == 1.0

    def test_state_under_triage_count(self, easy_env):
        easy_env.reset()
        gt = easy_env._cases[0]["ground_truth_esi"]
        if gt < 5:
            easy_env.step(TriageAction(esi_level=min(5, gt+2), disposition="waiting_room"))
            assert easy_env.state.under_triage_count == 1


# ──────────────────────────────────────────────────────────────
# Reward / scoring tests
# ──────────────────────────────────────────────────────────────

class TestReward:
    def test_perfect_episode_high_score(self, easy_env):
        easy_env.reset()
        for _ in range(10):
            easy_env.step(perfect_action(easy_env))
        score = easy_env.normalized_score()
        assert score >= 0.85  # perfect actions should score well

    def test_all_wrong_episode_low_score(self, easy_env):
        easy_env.reset()
        for i in range(10):
            gt = easy_env._cases[i]["ground_truth_esi"]
            easy_env.step(wrong_action(gt))
        score = easy_env.normalized_score()
        assert score < 0.30  # consistently wrong = low score

    def test_score_in_range(self, easy_env):
        easy_env.reset()
        for _ in range(10):
            easy_env.step(perfect_action(easy_env))
        score = easy_env.normalized_score()
        assert 0.0 <= score <= 1.0


# ──────────────────────────────────────────────────────────────
# All tasks complete
# ──────────────────────────────────────────────────────────────

class TestAllTasks:
    @pytest.mark.parametrize("task", ["easy", "medium", "hard"])
    def test_task_completes(self, task):
        env = ClinicalTriageEnvironment(task=task)
        obs = env.reset()
        step_count = 0
        while not obs.done:
            case = env._cases[env._queue_index]
            action = TriageAction(
                esi_level=case["ground_truth_esi"],
                resources_requested=case["ground_truth_resources"],
                disposition=case["ground_truth_disposition"],
                clinical_reasoning="Ground truth action.",
            )
            obs = env.step(action)
            step_count += 1
        assert obs.done
        assert step_count == 10

    @pytest.mark.parametrize("task", ["easy", "medium", "hard"])
    def test_task_score_range(self, task):
        env = ClinicalTriageEnvironment(task=task)
        env.reset()
        for i in range(10):
            case = env._cases[i]
            action = TriageAction(
                esi_level=case["ground_truth_esi"],
                resources_requested=case["ground_truth_resources"],
                disposition=case["ground_truth_disposition"],
            )
            env.step(action)
        score = env.normalized_score()
        assert 0.0 <= score <= 1.0

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            ClinicalTriageEnvironment(task="impossible")
