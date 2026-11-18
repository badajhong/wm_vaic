"""Frame budgets are invariant to simulator parallelism and survive resume."""

from fractions import Fraction

import pytest

from active_adaptation.learning.wm.schedule import FrameTrainingSchedule


def run_budget(chunks):
    schedule = FrameTrainingSchedule()
    counts = dict(wm=0, actor=0)
    for frames in chunks:
        schedule.accrue(frames)
        for kind, cost in (("wm", 1024 * 8), ("actor", 4096)):
            while schedule.available(kind, cost):
                schedule.consume(kind, cost)
                counts[kind] += 1
    return schedule, counts


def test_legacy_helper_budgets_match_supermimic_rates_independent_of_collection_partition():
    small, counts = run_budget([128 * 32] * 32)
    large, other_counts = run_budget([4096 * 32])
    assert counts == other_counts == {"wm": 224, "actor": 12}
    assert small.state_dict() == large.state_dict()
    assert large.wm_credit == 0
    assert large.actor_credit == Fraction(16384, 5)
    # Fractions are preserved, rather than silently rounding .4 starts/frame.
    assert counts["actor"] * 4096 + large.actor_credit == Fraction(2, 5) * 131072


@pytest.mark.parametrize("chunks", [
    [4096 * 32], [128 * 32] * 32, [512 * 32] * 8, [2048 * 32] * 2, [1, 4096 * 32 - 1],
])
def test_legacy_frame_budget_preserves_work_across_collection_partitions(chunks):
    from active_adaptation.learning.wm.algorithm import WMConfig

    cfg = WMConfig(schedule_mode="frames", critic_epochs=1, critic_num_minibatches=1)
    schedule = FrameTrainingSchedule(cfg.wm_predictions_per_frame, cfg.actor_starts_per_frame)
    costs = {"wm": cfg.wm_batch_size * cfg.wm_sequence_length, "actor": cfg.actor_batch_size}
    counts = {"wm": 0, "actor": 0}
    for frames in chunks:
        schedule.accrue(frames)
        for kind, cost in costs.items():
            while schedule.available(kind, cost):
                schedule.consume(kind, cost)
                counts[kind] += 1
    assert schedule.frames == 4096 * 32
    assert counts == {"wm": 16, "actor": 1}
    assert counts["actor"] * cfg.critic_epochs * cfg.critic_num_minibatches == 1
    assert schedule.wm_credit == schedule.actor_credit == 0


def test_legacy_frame_budget_has_one_actor_update_per_collection_without_deferred_credit():
    from active_adaptation.learning.wm.algorithm import WMConfig

    # Legacy frame budgets were defined for the 32-step PPO collection window.
    cfg = WMConfig(schedule_mode="frames", critic_epochs=1, critic_num_minibatches=1, train_every=32)
    schedule = FrameTrainingSchedule(cfg.wm_predictions_per_frame, cfg.actor_starts_per_frame)
    for _ in range(8):
        schedule.accrue(4096 * cfg.train_every)
        assert schedule.available("wm", cfg.wm_batch_size * cfg.wm_sequence_length) == 16
        assert schedule.available("actor", cfg.actor_batch_size) == 1
        assert cfg.warmup_model_updates <= 16
        for _ in range(16):
            schedule.consume("wm", cfg.wm_batch_size * cfg.wm_sequence_length)
        schedule.consume("actor", cfg.actor_batch_size)
        assert schedule.wm_credit == schedule.actor_credit == 0


def test_deferred_actor_credit_is_retained_and_checkpoint_restores_fraction_exactly():
    schedule = FrameTrainingSchedule(.3, .4)
    schedule.accrue(11)
    schedule.consume("wm", 3)
    assert schedule.wm_credit == Fraction(3, 10)
    assert schedule.actor_credit == Fraction(22, 5)
    restored = FrameTrainingSchedule(.3, .4)
    restored.load_state_dict(schedule.state_dict())
    for instance in (schedule, restored):
        instance.accrue(9)
        instance.consume("actor", 8)
    assert schedule.state_dict() == restored.state_dict()
    assert restored.actor_credit == 0 and restored.wm_credit == 3


def test_resume_rate_change_preserves_prior_sample_credit_and_legacy_has_no_historical_debt():
    schedule = FrameTrainingSchedule()
    schedule.accrue(3)
    restored = FrameTrainingSchedule(2., .1)
    restored.load_state_dict(schedule.state_dict())
    restored.accrue(2)
    assert restored.wm_credit == 46 and restored.actor_credit == Fraction(7, 5)
    restored.load_state_dict(None)
    assert restored.frames == restored.wm_credit == restored.actor_credit == 0


@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf")])
def test_invalid_rates_rejected(value):
    with pytest.raises(ValueError, match="finite and positive"):
        FrameTrainingSchedule(value, .4)


def test_budget_cannot_be_overdrawn_or_accrue_fractional_frames():
    schedule = FrameTrainingSchedule()
    with pytest.raises(ValueError, match="Insufficient"):
        schedule.consume("actor", 1)
    with pytest.raises(ValueError, match="nonnegative integer"):
        schedule.accrue(1.5)
