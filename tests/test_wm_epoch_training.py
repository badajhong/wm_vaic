"""Full CPU learner verifies epoch scheduling with real optimizer steps."""

from collections import Counter
import copy

import pytest
import torch
from torchrl.envs.transforms import VecNorm

from test_wm_training import make_policy, make_rollout, assert_tree_equal


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(413)
    yield
    torch.set_num_threads(previous)


def epoch_policy(**overrides):
    settings = dict(schedule_mode="epochs", train_every=32, wm_epochs=3, wm_num_minibatches=16,
                    wm_batch_size=1, wm_sequence_length=8, imagination_horizon=16,
                    ppo_epochs=3, num_minibatches=8, actor_batch_size=8,
                    actor_microbatch_size=1, critic_epochs=3, critic_num_minibatches=8,
                    critic_microbatch_size=16, replay_capacity=256, validation_capacity=128,
                    require_model_validation=False, warmup_model_updates=10,
                    actor_lr=1e-7, actor_lr_max=1e-7, encoder_lr=1e-7,
                    actor_drift_batch_size=0)
    settings.update(overrides)
    return make_policy(**settings)


def _identity(state):
    return tuple(state.detach().reshape(-1).tolist())


def test_epoch_loop_executes_48_model_24_actor_24_critic_steps_and_reuses_fixed_pools(monkeypatch):
    policy = epoch_policy()
    rollout = make_rollout(policy)
    normalizer = VecNorm(in_keys=["policy"], decay=0.)
    normalizer(rollout[:, 0].clone())
    policy.set_observation_normalizer(normalizer, "train")
    original_policy = copy.deepcopy(policy.actor.state_dict())
    students = {name: copy.deepcopy(getattr(policy, name).state_dict()) for name in policy.student_module_names
                if hasattr(policy, name)}
    counts = dict(model=0, actor=0, critic=0)
    order, model_rows, actor_rows = [], [], []
    for name, optimizer in (("model", policy.opt_world_model), ("actor", policy.opt_policy), ("critic", policy.opt_critic)):
        original = optimizer.step
        def step(*args, _original=original, _name=name, **kwargs):
            counts[_name] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(optimizer, "step", step)
    original_model, original_actor = policy._fit_world_model, policy._imagine
    def fit_model(batch):
        order.append("model")
        model_rows.append(_identity(batch["state"][:, 0]))
        assert batch.batch_size == (1, 8)
        assert batch.device.type == "cpu"
        return original_model(batch)
    def imagine(starts, horizon):
        order.append("actor")
        actor_rows.append(_identity(starts["state"]))
        assert starts.batch_size == (1,) and horizon == 16
        assert policy._oracle_checked
        assert not starts["is_init"].any()
        return original_actor(starts, horizon)
    def validate():
        order.append("validation")
        assert counts == dict(model=16, actor=0, critic=0)
        assert policy._optimizer_model_updates_since_resume == 16
        policy._validated_horizon = 2
        policy._oracle_checked = True
        return {"validation/approved_horizon": 2}
    monkeypatch.setattr(policy, "_fit_world_model", fit_model)
    monkeypatch.setattr(policy, "_imagine", imagine)
    monkeypatch.setattr(policy, "_validate_world_model", validate)
    def forbidden(*args, **kwargs):
        raise AssertionError("Epoch mode must use deterministic BPTT and its same-rollout critic")
    for method in ("_update_ppo", "_compute_advantage"):
        monkeypatch.setattr(policy, method, forbidden)
    info = policy.train_op(rollout)
    assert counts == dict(model=48, actor=24, critic=24)
    assert order == (["model"] * 16 + ["validation"] + ["actor"] * 8
                     + (["model"] * 16 + ["actor"] * 8) * 2)
    for epoch in (1, 2):
        assert Counter(model_rows[epoch * 16:(epoch + 1) * 16]) == Counter(model_rows[:16])
        assert Counter(actor_rows[epoch * 8:(epoch + 1) * 8]) == Counter(actor_rows[:8])
    assert model_rows[:16] != model_rows[16:32]
    assert actor_rows[:8] != actor_rows[8:16]
    assert info["wm/model_updates"] == info["wm/planned_model_updates"] == 48
    assert info["wm/actor_updates"] == info["wm/actor_update_attempts"] == info["critic/updates"] == 24
    assert info["wm/model_prediction_samples"] == 48 * 8
    assert info["wm/actor_starts"] == info["wm/accepted_actor_starts"] == 24
    assert info["wm/actor_imagined_steps"] == 24 * 16
    assert info["wm/actor_minibatch_size"] == 1
    assert info["wm/epoch_schedule"] == 1 and info["wm/frame_schedule"] == 0
    assert len(policy.replay) == 96 and len(policy.validation_replay) == 32
    assert policy.world_model.state_stats.count == 2 * 96
    torch.testing.assert_close(policy._norm_snapshot["policy"][0], normalizer.loc["policy"])
    assert any(not torch.equal(value, original_policy[key]) for key, value in policy.actor.state_dict().items())
    for name, before in students.items():
        assert_tree_equal(getattr(policy, name).state_dict(), before)


def test_epoch_resume_rebuilds_replay_and_oracle_before_policy_and_holds_validation_separate(monkeypatch):
    source = epoch_policy()
    source.train_op(make_rollout(source))
    restored = epoch_policy()
    restored.load_state_dict(copy.deepcopy(source.state_dict()))
    assert len(restored.replay) == len(restored.validation_replay) == 0
    assert restored._optimizer_model_updates_since_resume == 0
    assert restored._oracle_checked is False
    rollout = make_rollout(restored)
    actual_validation = []
    def validate():
        actual_validation.append(restored._optimizer_model_updates_since_resume)
        # This flag intentionally remains false: passing a stale saved gate is
        # insufficient to start BPTT after replay has been reconstructed.
        return {"validation/approved_horizon": 0}
    monkeypatch.setattr(restored, "_validate_world_model", validate)
    before_actor = copy.deepcopy(restored.actor.state_dict())
    info = restored.train_op(rollout)
    assert actual_validation == [16]
    assert info["wm/model_updates"] == 48
    assert info["wm/actor_update_attempts"] == info["critic/updates"] == 0
    assert info["wm/policy_ready"] == 0
    heldout = restored.validation_replay.sample(1, 32, draws=torch.zeros(1, dtype=torch.long))
    expected = restored.codec.pack(rollout["wm_observation"][0:1])
    torch.testing.assert_close(heldout["state"], expected, rtol=0, atol=0)
    train = restored.replay.sample(3, 32, draws=torch.arange(3))
    torch.testing.assert_close(train["state"], restored.codec.pack(rollout["wm_observation"][1:]), rtol=0, atol=0)
    # Collection exploration scheduling may refresh actor_std independently.
    for key, value in restored.actor.state_dict().items():
        if not key.endswith("actor_std"):
            torch.testing.assert_close(value, before_actor[key], rtol=0, atol=0)


def test_epoch_warmup_can_delay_first_policy_epoch_without_drawing_new_actor_pool(monkeypatch):
    policy = epoch_policy(warmup_model_updates=20)
    model_updates_at_actor = []
    original = policy._imagine
    def imagine(starts, horizon):
        model_updates_at_actor.append(policy._optimizer_model_updates_since_resume)
        return original(starts, horizon)
    monkeypatch.setattr(policy, "_imagine", imagine)
    info = policy.train_op(make_rollout(policy))
    assert info["wm/model_updates"] == 48
    assert info["wm/actor_update_attempts"] == info["critic/updates"] == 16
    assert model_updates_at_actor == [32] * 8 + [48] * 8
