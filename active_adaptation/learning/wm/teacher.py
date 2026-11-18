"""WM teacher that consumes privileged observations without a latent encoder.

PPOVEL remains unchanged. Call ``configure_raw_teacher`` immediately after its
constructor, before creating the WM actor controller or loading a checkpoint.
Privileged vectors enter the actor MLP directly. Fixed object geometry and its
point-cloud transform are excluded; only the optional height CNN remains as a
learned feature extractor outside the actor.
"""

import copy
from collections.abc import Mapping, Sequence

import torch
from torch import nn
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq
from torchrl.modules import ProbabilisticActor

from ..modules.distributions import IndependentNormal
from ..ppo.common import ACTION_KEY, Actor, CatTensors, make_mlp
from ..ppo.ppo_vel import OBJECT_GEO_KEY, OBJECT_TRANS_KEY, PRIV_FEATURE_KEY


class _ObjectPrefix(nn.Module):
    """An actor-only view; the source object observation remains untouched."""

    def __init__(self, stop):
        super().__init__()
        self.stop = stop

    def forward(self, value):
        return value[..., :self.stop]


def _object_prefix_without_duplicate_pose(policy, observation_spec):
    """Prove the trailing 12 raw measurements also occur in ``priv``.

    The known observation classes have widths 2/2/3*contacts/3/9. Unknown
    layouts or options retain the complete object vector. Privileged values
    still use their existing normalization; their raw measurements duplicate
    the omitted, unnormalized object pose.
    """
    config = getattr(getattr(policy, "env", None), "cfg", None)
    observations = getattr(config, "observation", None)
    if not isinstance(observations, Mapping):
        return None
    objects, privileged = observations.get("object_"), observations.get("priv")
    if not isinstance(objects, Mapping) or not isinstance(privileged, Mapping):
        return None
    expected_order = ("object_xy_b", "object_heading_b", "ref_contact_pos_b",
                      "object_pos_b", "object_ori_b")
    if tuple(objects) != expected_order:
        return None
    for name in expected_order[-2:]:
        # These two classes accept no observation options. Refuse even matching
        # nonempty options rather than assuming a future class remains noiseless.
        if name not in privileged or objects[name] not in (None, {}) or privileged[name] not in (None, {}):
            return None
    allowed_prefix_options = ({"noise_std", "episodic_noise_std"},
                              {"noise_std", "episodic_noise_std"},
                              {"noise_std", "episodic_noise_std", "yaw_only"})
    for name, allowed in zip(expected_order[:3], allowed_prefix_options):
        options = objects[name]
        if options is not None and (not isinstance(options, Mapping) or not set(options) <= allowed):
            return None
    command = getattr(config, "command", None)
    contacts = command.get("contact_eef_body_name") if isinstance(command, Mapping) else None
    if not isinstance(contacts, Sequence) or isinstance(contacts, str) or not contacts:
        return None
    prefix_width = 4 + 3 * len(contacts)
    if ("object_" not in observation_spec or "priv" not in observation_spec
            or observation_spec["object_"].shape[-1] != prefix_width + 12
            or observation_spec["priv"].shape[-1] < 12):
        return None
    return prefix_width


def configure_raw_teacher(policy, observation_spec):
    """Replace the initialized PPO teacher with a freshly initialized raw MLP.

    The actor keeps its 512/256/256 hidden widths, Gaussian head and reference
    joint-position residual. No latent-teacher weights are converted. The
    critic, height CNN, object transform and retained student modules keep their
    existing parameters. The replacement Adam owns only actor and optional
    height parameters. Return the batch-size-independent checkpoint schema.
    """
    if policy.cfg.phase != "train":
        raise ValueError("The raw WM teacher requires phase=train")
    if not hasattr(policy, "encoder_priv"):
        raise ValueError("Configure the raw WM teacher once, immediately after PPO initialization")
    old_actor = policy.actor.module[0]
    actor_keys = tuple(old_actor[0].in_keys)
    if actor_keys[-1:] != (PRIV_FEATURE_KEY,) or len(actor_keys) != 3:
        raise ValueError("Unexpected PPO teacher inputs; raw-input construction requires command, policy, priv_feature")
    # A single fixed object point cloud repeats shape information already fixed
    # by the task. Exclude both raw and transformed coordinates, so the raw
    # teacher needs no point-cloud transform during collection or imagination.
    raw_keys = tuple(key for key in policy.encoder_priv[0].in_keys
                     if key not in (OBJECT_GEO_KEY, OBJECT_TRANS_KEY))
    projection = None
    if "priv" in raw_keys and "object_" in raw_keys:
        prefix = _object_prefix_without_duplicate_pose(policy, observation_spec)
        if prefix is not None:
            projection = Mod(_ObjectPrefix(prefix), ["object_"], ["_actor_object"])
            raw_keys = tuple("_actor_object" if key == "object_" else key for key in raw_keys)
    input_keys = (*actor_keys[:-1], *raw_keys)
    if len(input_keys) != len(set(input_keys)):
        raise ValueError("Raw WM teacher input fields must be unique")
    if len(old_actor) != 4:
        raise ValueError("Unexpected PPO teacher structure; its reference-action module must be preserved")

    # Preserve the existing residual equation, including disabled-residual mode.
    # That module has no trained weights; the entire actor MLP/head starts fresh.
    residual = copy.deepcopy(old_actor[3])
    if any(True for _ in residual.parameters()):
        raise ValueError("The reference-action residual must not contain learned parameters")
    actor = ProbabilisticActor(
        module=Seq(
            CatTensors(list(input_keys), "_actor_inp", del_keys=False, sort=False),
            Mod(make_mlp(list(getattr(policy.cfg, "actor_hidden_dims", (512, 256, 256)))),
                ["_actor_inp"], ["_actor_feature"]),
            Mod(Actor(policy.action_dim, init_noise_scale=policy.cfg.init_noise_scale,
                      load_noise_scale=policy.cfg.load_noise_scale),
                ["_actor_feature"], IndependentNormal.dist_keys),
            residual,
        ),
        in_keys=IndependentNormal.dist_keys,
        out_keys=[ACTION_KEY],
        distribution_class=IndependentNormal,
        return_log_prob=True,
    ).to(policy.device)
    with torch.no_grad():
        sample = observation_spec.zero().to(policy.device)
        if hasattr(policy, "height_encoder"):
            policy.height_encoder(sample)
        if projection is not None:
            projection(sample)
        # Initialize lazy dimensions without sampling an action.
        actor.get_dist(sample)
        # Standard-gain hidden layers; only the action head starts near zero, so
        # the initial policy still equals the reference-action residual. A 0.01
        # gain on LayerNorm'd hidden layers leaves their output unchanged but
        # makes each fixed-size Adam step ~100x larger relative to the weights.
        head = actor.module[0][2].module.actor_mean
        for module in actor.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, .01 if module is head else 2 ** .5)
                nn.init.zeros_(module.bias)
    schema = (
        "raw_teacher_input_v1",
        tuple((key, sample[key].shape[-1]) for key in input_keys),
        policy.action_dim,
        bool(policy.cfg.enable_residual_distillation),
    )
    actor.module[0][2].module.actor_std.requires_grad_(False)
    groups = [{"params": actor.parameters(), "name": "actor"}]
    if hasattr(policy, "height_cnn"):
        groups.append({"params": policy.height_cnn.parameters(), "name": "height"})
    optimizer = torch.optim.Adam(groups, lr=policy.cfg.lr)

    policy.actor = actor
    del policy.encoder_priv
    policy.opt_policy = optimizer
    if projection is not None:
        policy.raw_teacher_projection = projection
    policy.teacher_input_schema = schema
    return schema


def teacher_dist(policy, observations):
    """Evaluate raw teacher inputs after the caller's normalizer snapshot."""
    if hasattr(policy, "height_encoder"):
        policy.height_encoder(observations)
    if hasattr(policy, "raw_teacher_projection"):
        policy.raw_teacher_projection(observations)
    return policy.actor.get_dist(observations)


def raw_teacher_rollout_policy(policy):
    """Collection/evaluation teacher without student or recurrent execution."""
    modules = []
    if hasattr(policy, "height_encoder"):
        modules.append(policy.height_encoder)
    if hasattr(policy, "raw_teacher_projection"):
        modules.append(policy.raw_teacher_projection)
    modules.append(policy.actor)
    return Seq(*modules, selected_out_keys=["sample_log_prob", ACTION_KEY, *IndependentNormal.dist_keys])
