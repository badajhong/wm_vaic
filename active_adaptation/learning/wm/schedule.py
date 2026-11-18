"""Replay budgets expressed in actual collected transitions, not env count."""

from fractions import Fraction
import math


class FrameTrainingSchedule:
    """Keep exact fractional sample credit across collection batches/resumes.

    Credit is consumed only when a corresponding optimizer attempt is made.
    Deferred actor work (warmup or an explicitly enabled accuracy gate) remains
    visible credit, rather than silently changing the requested replay ratio.
    Rates may be changed on resume; saved credit remains in sample units.
    """

    def __init__(self, wm_predictions_per_frame=14., actor_starts_per_frame=.4):
        for value in (wm_predictions_per_frame, actor_starts_per_frame):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("WM prediction and actor-start budgets must be finite and positive")
        self.wm_rate = Fraction(str(wm_predictions_per_frame))
        self.actor_rate = Fraction(str(actor_starts_per_frame))
        self.wm_credit = Fraction(0)
        self.actor_credit = Fraction(0)
        self.frames = 0

    def accrue(self, frames):
        if isinstance(frames, bool) or int(frames) != frames or frames < 0:
            raise ValueError("Collected frame count must be a nonnegative integer")
        self.frames += int(frames)
        self.wm_credit += int(frames) * self.wm_rate
        self.actor_credit += int(frames) * self.actor_rate

    def available(self, kind, samples_per_update):
        if samples_per_update < 1:
            raise ValueError("An optimizer update must consume at least one sample")
        return int(getattr(self, kind + "_credit") // samples_per_update)

    def consume(self, kind, samples_per_update):
        if self.available(kind, samples_per_update) < 1:
            raise ValueError(f"Insufficient {kind} replay budget")
        setattr(self, kind + "_credit", getattr(self, kind + "_credit") - samples_per_update)

    def state_dict(self):
        encode = lambda value: (value.numerator, value.denominator)
        return {"version": 1, "frames": self.frames,
                "wm_credit": encode(self.wm_credit), "actor_credit": encode(self.actor_credit)}

    def load_state_dict(self, state):
        # Older checkpoints did not account for real-frame budgets. Do not
        # invent a debt for their entire past training history.
        if state is None:
            self.wm_credit = self.actor_credit = Fraction(0)
            self.frames = 0
            return
        if state.get("version") != 1:
            raise ValueError("Unsupported WM frame-budget checkpoint version")
        wm_credit, actor_credit = (Fraction(*state[key]) for key in ("wm_credit", "actor_credit"))
        if min(wm_credit, actor_credit) < 0 or state["frames"] < 0:
            raise ValueError("Negative WM frame budget in checkpoint")
        self.wm_credit, self.actor_credit = wm_credit, actor_credit
        self.frames = int(state["frames"])
