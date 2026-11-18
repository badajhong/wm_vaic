"""Simulator-free stand-in world model for learner-mechanics tests.

Production WM training uses PhysicalWorldModel and GroundedWorldModel. This
generic observation model exists only so learner tests can run on synthetic
transitions; it is not a fallback for simulator dynamics or reward equations.
"""

import torch
from torch import nn
from torch.nn import functional as F

from active_adaptation.learning.wm.core import RunningMoments


class DifferentiableWorldModel(nn.Module):
    """Residual observation dynamics and task reward/termination heads.

    All heads share an MLP conditioned on observation and action. Static object
    geometry is copied exactly; supplied next-reference inputs are teacher
    forced. Without ``next_exogenous``, reference inputs retain their current
    value, so callers should supply the reference trajectory during imagination.
    Reward channels retain the environment's reward-group ordering.
    """

    def __init__(self, codec, action_dim, reward_dim, hidden_dims=(512, 512), device="cpu"):
        super().__init__()
        self.codec = codec
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.state_stats = RunningMoments(codec.state_dim)
        self.action_stats = RunningMoments(action_dim)
        self.reward_stats = RunningMoments(reward_dim)
        layers = []
        in_dim = codec.state_dim + action_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(in_dim, width), nn.LayerNorm(width), nn.SiLU()))
            in_dim = width
        self.trunk = nn.Sequential(*layers)
        self.dynamics_head = nn.Linear(in_dim, codec.state_dim)
        self.reward_head = nn.Linear(in_dim, reward_dim)
        self.termination_head = nn.Linear(in_dim, 1)
        self.discount_head = nn.Linear(in_dim, 1)
        for head in (self.dynamics_head, self.reward_head):
            nn.init.normal_(head.weight, std=1e-3)
            nn.init.zeros_(head.bias)
        nn.init.constant_(self.termination_head.bias, -2.0)
        nn.init.constant_(self.discount_head.bias, 4.0)
        dynamic_mask = torch.zeros(codec.state_dim)
        exogenous_mask = torch.zeros(codec.state_dim, dtype=torch.bool)
        for key in codec.dynamic_keys:
            dynamic_mask[codec.slices[key]] = 1
        for key in codec.exogenous_keys:
            exogenous_mask[codec.slices[key]] = True
        self.register_buffer("dynamic_mask", dynamic_mask)
        self.register_buffer("exogenous_mask", exogenous_mask)
        self.to(device)

    @torch.no_grad()
    def update_stats(self, state, action, next_state, reward, valid=None):
        if valid is not None:
            state, action, next_state, reward = (value[valid] for value in (state, action, next_state, reward))
        self.state_stats.update(state)
        self.state_stats.update(next_state)
        self.action_stats.update(action)
        self.reward_stats.update(reward)

    def _predict(self, state, action, next_exogenous=None):
        features = self.trunk(torch.cat((
            self.state_stats.normalize(state), self.action_stats.normalize(action)
        ), dim=-1))
        delta = self.dynamics_head(features) * self.state_stats.std
        next_state = state + delta * self.dynamic_mask
        if next_exogenous is not None:
            if next_exogenous.shape != state.shape:
                raise ValueError("next_exogenous must be a packed vector matching state shape")
            next_state = torch.where(self.exogenous_mask, next_exogenous, next_state)
        reward_normalized = self.reward_head(features)
        termination_logit = self.termination_head(features)
        discount_logit = self.discount_head(features)
        return next_state, reward_normalized, termination_logit, discount_logit

    def forward(self, state, action, next_exogenous=None):
        next_state, reward, termination_logit, discount_logit = self._predict(
            state, action, next_exogenous
        )
        return (
            next_state,
            self.reward_stats.denormalize(reward),
            termination_logit.sigmoid(),
            discount_logit.sigmoid(),
        )

    def transition_loss(
        self, state, action, next_state, reward, terminated, discount, group_weights=None,
        return_prediction=False,
    ):
        """Return unreduced losses of shape ``state.shape[:-1]``.

        Dynamics averages each group's normalized feature errors before averaging
        groups, preventing a large height map from overwhelming object dynamics.
        ``group_weights`` optionally changes those group weights. Callers must
        mask transitions whose returned next observation is an automatic reset.
        Termination and discount targets must each be one scalar per transition.
        """
        predicted, reward_normalized, termination_logit, discount_logit = self._predict(
            state, action, next_exogenous=next_state
        )
        error = ((predicted - next_state) / self.state_stats.std).square()
        group_weights = group_weights or {}
        losses = {}
        total_weight = 0.0
        dynamics = torch.zeros_like(error[..., 0])
        for key in self.codec.dynamic_keys:
            feature_mask = self.dynamic_mask[self.codec.slices[key]]
            group_loss = (error[..., self.codec.slices[key]] * feature_mask).sum(-1) / feature_mask.sum().clamp_min(1)
            losses[f"dynamics/{key}"] = group_loss
            weight = float(group_weights.get(key, 1.0))
            if weight < 0:
                raise ValueError("Dynamics group weights must be nonnegative")
            dynamics = dynamics + weight * group_loss
            total_weight += weight
        if total_weight <= 0:
            raise ValueError("At least one dynamics group weight must be positive")
        losses["dynamics"] = dynamics / total_weight
        losses["reward"] = (
            reward_normalized - self.reward_stats.normalize(reward)
        ).square().mean(-1)
        for name, logits, target in (
            ("termination", termination_logit, terminated),
            ("discount", discount_logit, discount),
        ):
            target = target.to(logits).reshape(logits.shape)
            losses[name] = F.binary_cross_entropy_with_logits(logits, target, reduction="none").squeeze(-1)
        return (losses, predicted) if return_prediction else losses

