"""Compose physical dynamics with the task's observation and reward equations."""

import torch
from torch import nn


class GroundedWorldModel(nn.Module):
    def __init__(self, physics, adapter, reward):
        super().__init__()
        self.physics = physics
        self.adapter = adapter
        self.reward_model = reward
        # Execution-only switch for paired profiling; no checkpoint keys or
        # task/model configuration change. Minimal test adapters may only
        # implement the public rebuild API.
        self.use_fused_finish = hasattr(adapter, "prepare_context")
        # Parameter-free helpers are cached by rollout horizon. Their recurrent
        # probability state belongs to each imagined rollout, never to replay.
        self._soft_termination_models = {}
        # First-contact timing is used only by these supported task equations.
        # Other sensor bodies still contribute to force/auxiliary validation,
        # but unused wrist timers must not dominate ankle reward validation.
        timer_kinds = {"impact_force_l2", "feet_air_time_skateboard", "feet_slip"}
        body_ids = sorted({int(index) for term in getattr(reward, "reward_terms", ())
                           if term.kind in timer_kinds
                           for index in torch.as_tensor(term.body_ids).reshape(-1).tolist()})
        self.register_buffer("reward_contact_body_ids", torch.tensor(body_ids, dtype=torch.long), persistent=False)
        # Without contact prediction, groups whose exact reward reads contact
        # force/timers are excluded from the factual reward loss: their error
        # cannot be reduced by better kinematics and would only distort them.
        self.contact_model = bool(getattr(physics, "contact_model", True))
        groups = None
        if not self.contact_model and hasattr(reward, "contact_free_groups"):
            groups = reward.contact_free_groups()
        # This composite is not moved as a whole; place the mask with the physics
        # weights, since a captured WM update cannot copy it from the host.
        parameters = physics.parameters() if isinstance(physics, nn.Module) else ()
        device = next((p.device for p in parameters), torch.device("cpu"))
        self.register_buffer("reward_loss_group_mask",
                             None if groups is None else torch.tensor(groups, dtype=torch.float32, device=device),
                             persistent=False)
        if not self.contact_model and hasattr(adapter, "prepare_context"):
            adapter.rebuild_contact_forces = False

    def _reward_options(self, kinematic_reward):
        options = {"kinematic": kinematic_reward}
        if not self.contact_model:
            # Exact contact terms cannot be evaluated without contact prediction;
            # their groups are already excluded from the factual reward loss.
            options["skip_contact"] = True
        return options

    @property
    def state_stats(self):
        return self.physics.state_stats

    def update_stats(self, state, action, next_state, reward, valid=None):
        return self.physics.update_stats(state, action, next_state, reward, valid=valid)

    def _finish(self, state, physical, action, reference, *, rebuild_observations=True,
                kinematic_reward=False):
        if self.use_fused_finish:
            previous_context = self.adapter._get_contexts(state)
            reference_context = (self.adapter._get_contexts(reference)
                                 if reference is not None else None)
            context = self.adapter.prepare_context(
                physical, reference, action, previous_state=state,
                previous_context=previous_context, template_context=reference_context)
            context, terminated = self.reward_model.update_termination_state(previous_context, context)
            reward, _, discount = self.reward_model(
                previous_context, context, action, reference_context, compute_termination=False,
                **self._reward_options(kinematic_reward))
            prediction = self.adapter.pack_observations(
                physical, reference, context, rebuild_observations=rebuild_observations)
            return prediction, reward, terminated.float(), discount
        # Retain the public composition for paired execution-parity checks.
        prediction = self.adapter.rebuild(physical, reference, action, previous_state=state,
                                          rebuild_observations=rebuild_observations)
        prediction, terminated = self.reward_model.update_termination_state(state, prediction)
        reward, _, discount = self.reward_model(state, prediction, action, reference,
                                                compute_termination=False, **self._reward_options(kinematic_reward))
        return prediction, reward, terminated.float(), discount

    def forward(self, state, action, next_exogenous, *, kinematic_reward=False):
        physical = self.physics(state, action, next_exogenous)
        return self._finish(state, physical, action, next_exogenous, kinematic_reward=kinematic_reward)

    def set_kinematic_imagination_reward(self, enabled):
        """Imagined actor rewards use only functions of the predicted kinematics.

        Factual model fitting, validation and the hard ``forward`` keep the exact
        task terms; only :meth:`imagine` and :meth:`imagine_soft` are affected.
        """
        if not isinstance(enabled, bool):
            raise ValueError("kinematic imagination reward switch must be a bool")
        if not enabled and not self.contact_model:
            raise ValueError("A world model without contact prediction (algo.wm_contact_model=false) "
                             "requires the kinematic imagination reward (algo.imagination_kinematic_reward=true)")
        self.kinematic_imagination_reward = enabled

    def _imagined_reward_options(self):
        # Pass the keyword only when enabled, so subclasses overriding the
        # three-argument forward keep working with exact imagined rewards.
        return {"kinematic_reward": True} if getattr(self, "kinematic_imagination_reward", False) else {}

    def set_imagination_reference(self, provider):
        """Attach immutable motion tensors without changing factual WM fitting."""
        self.imagination_reference = provider

    def imagine(self, state, action):
        """Hard-equation rollout for diagnostics; actor BPTT uses imagine_soft."""
        if not hasattr(self, "imagination_reference"):
            raise RuntimeError("Independent imagination requires a VAICImaginationReference provider")
        reference, truncated = self.imagination_reference.step(state, resample_noise=True)
        prediction, reward, terminated, discount = self(state, action, reference, **self._imagined_reward_options())
        return prediction, reward, terminated, discount, truncated

    def begin_soft_imagination(self, state, horizon):
        """Seed a separate differentiable failure process from logged counters."""
        from .termination import SoftTerminationModel
        if horizon not in self._soft_termination_models:
            self._soft_termination_models[horizon] = SoftTerminationModel(
                self.reward_model, horizon, contact_force=getattr(self, "soft_contact_force", True))
        model = self._soft_termination_models[horizon]
        return model, model.initialize(state)

    def set_soft_contact_force(self, enabled):
        """Choose whether imagined lost-contact also requires predicted force."""
        if not isinstance(enabled, bool):
            raise ValueError("soft contact-force switch must be a bool")
        if enabled and not self.contact_model:
            raise ValueError("A world model without contact prediction (algo.wm_contact_model=false) cannot "
                             "judge imagined lost contact by force (algo.imagination_contact_force_termination=false)")
        self.soft_contact_force = enabled
        self._soft_termination_models.clear()

    def imagine_soft(self, state, action, termination_model, termination_state):
        """Use soft failure hazard for policy learning, preserving task equations.

        Physical predictions still carry the exact analytic counters. The soft
        consecutive-violation distribution advances separately and is never
        reset from those hard counters during a rollout. Motion/time truncation
        remains a distinct, hard bootstrap boundary.
        """
        if not hasattr(self, "imagination_reference"):
            raise RuntimeError("Independent imagination requires a VAICImaginationReference provider")
        reference, truncated = self.imagination_reference.step(state, resample_noise=True)
        prediction, reward, _, discount = self(state, action, reference, **self._imagined_reward_options())
        hazard, termination_state = termination_model.step(
            state, prediction, termination_state, next_reference=reference)
        return prediction, reward, hazard, discount, truncated, termination_state

    def baseline_prediction(self, state, action, next_exogenous):
        physical = self.physics.baseline_prediction(state, action, next_exogenous)
        return self._finish(state, physical, action, next_exogenous)[0]

    def validation_errors(self, prediction, target):
        errors = self.physics.validation_errors(prediction, target)
        if "contact_event_error" not in errors:
            return errors
        for key in tuple(errors):
            if key.startswith("contact_event_"):
                errors[f"all_sensor_{key}"] = errors.pop(key)
        if self.reward_contact_body_ids.numel():
            indices = self.reward_contact_body_ids.to(prediction.device)
            def first(state):
                duration = self.physics.context(state, "contact.current_contact_time").index_select(-1, indices)
                return (duration > 0) & (duration < self.physics.step_dt + 1e-8)
            predicted, actual = first(prediction), first(target)
            errors["contact_event_error"] = (predicted != actual).float().mean(-1)
            errors["contact_event_true_count"] = actual.sum(-1).float()
            errors["contact_event_miss_count"] = (actual & ~predicted).sum(-1).float()
        return errors

    def transition_loss(self, state, action, next_state, reward, terminated, discount,
                        group_weights=None, return_prediction=False):
        losses, physical = self.physics.transition_loss(
            state, action, next_state, reward, terminated, discount, return_prediction=True)
        prediction, predicted_reward, _, _ = self._finish(state, physical, action, next_state,
                                                         rebuild_observations=False)
        # The reward is an equation of the predicted state. This auxiliary loss
        # improves its relevant physics without introducing a reward head.
        error = (predicted_reward - reward).square()
        mask = self.reward_loss_group_mask
        if mask is None:
            losses["reward"] = error.mean(-1)
        else:
            losses["reward"] = (error * mask).sum(-1) / mask.sum().clamp_min(1)
        if return_prediction:
            return losses, prediction
        return losses
