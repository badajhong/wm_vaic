# VAIC: Vision-Guided Humanoid Agile Object Interaction Control via Decoupled Commands

<div align="center">
<a href="https://vaic-humanoid.github.io/">
	<img alt="Website" src="https://img.shields.io/badge/Website-Visit-blue?style=flat&logo=google-chrome"/>
</a>

<a href="https://arxiv.org/abs/2606.09286">
	<img alt="Arxiv" src="https://img.shields.io/badge/Paper-Arxiv-b31b1b?style=flat&logo=arxiv"/>
</a>

<a href="https://github.com/ldt29/VAIC/stargazers">
	<img alt="GitHub stars" src="https://img.shields.io/github/stars/ldt29/VAIC?style=social"/>
</a>

</div>

This repository hosts the open-source release for the paper VAIC: Vision-Guided Humanoid Agile Object Interaction Control via Decoupled Commands.


## 🚀 Quick Start

```bash
# setup conda environment
conda create -n vaic python=3.11 -y
conda activate vaic

# install isaacsim
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
isaacsim # test isaacsim

# install isaaclab
cd ..
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.2
./isaaclab.sh -i none

# install vaic
cd ..
git clone https://github.com/ldt29/VAIC
cd VAIC
pip install -e .
```

## Verify Your Data
Visualize motions in Isaac Sim with `task.command.replay_motion=true`:

```bash
python scripts/play.py algo=ppo_vel_train task=G1/vaic/skateboard_tea task.command.replay_motion=true
```


## Train and Evaluate

Teacher policy

```bash
# train policy
python scripts/train.py algo=ppo_vel_train task=G1/vaic/skateboard_tea
# evaluate policy
python scripts/play.py algo=ppo_vel_train task=G1/vaic/skateboard_tea checkpoint_path=run:<wandb-run-path>
```

Student policy

```bash
# train policy
python scripts/train.py algo=ppo_vel_finetune task=G1/vaic/skateboard_stu checkpoint_path=run:<student_wandb-run-path>
# evaluate policy
python scripts/play.py algo=ppo_vel_finetune task=G1/vaic/skateboard_stu checkpoint_path=run:<student_wandb-run-path>
```
To export trained policies, add `export_policy=true` to the play script.


## World-model teacher training

Use the existing `vaic` conda environment:

```bash
conda activate vaic
python scripts/train.py algo=wm_train task=G1/vaic/skateboard_tea
```

This command alone runs the tuned WM recipe. Its defaults:

| Setting | Default |
|---|---|
| `algo.train_every` | 16 real steps per iteration (PPO keeps 32) |
| `algo.init_noise_scale`, `algo.exploration_noise_end` | 0.05, 0.05 |
| `algo.replay_capacity` | 524288 |
| `algo.actor_microbatch_size` | 1024 |
| `algo.wm_loss_scales.object_root_velocity` | 1.0 (board twist weighted 6.25x the robot's) |

`algo.task_defaults` holds the WM training-only task settings. `scripts/train.py`
merges them into `task` before the environment is built:

- `cum_body_pos_error`, `cum_body_ori_error` and their `_local` variants use
  `min_steps=2` (strict).
- `cum_lost_contact_steps` uses `frc_thres=0.0` and `min_steps=8`.
- `command.reset_sampling=return`.

An explicit `task.*` override on the command line wins over these. `play.py`
and PPO keep the task file's values, so evaluation still sets its own
termination, for example `min_steps=8`.

`wm_train` keeps the `ppo_vel_train` actor hidden widths and critic, observation groups,
reference-action residual, observation normalization,
task rewards, terminations, domain randomization, motion sampling, and experiment
defaults. World-model context is used only during training; it is never appended to
actor or critic inputs. The run is named with `wm` to distinguish it from PPO runs.
The entry points import this checkout even if the environment's editable install
points to another VAIC repository.

All world-model code lives in the `active_adaptation/learning/wm/` package:

| Module | Role |
|---|---|
| `algorithm.py` | `WM` learner and `WMConfig` (exported as `active_adaptation.learning.wm.WM`) |
| `physics.py` | `PhysicalWorldModel`: learned dynamics, exact actuator/integration/FK |
| `grounded.py` | Composes the dynamics with the task observation/reward equations |
| `adapter.py`, `context.py` | Rebuild observations from the simulator context |
| `reward.py`, `termination.py` | Analytic task rewards and soft failure hazard |
| `imagination.py` | Independent motion-reference provider for imagined rollouts |
| `teacher.py` | Raw-input teacher actor |
| `replay.py`, `transport.py`, `epoch.py` | Replay storage, host-to-GPU transfer, epoch sampling |
| `actor_step.py`, `schedule.py`, `execution.py` | Actor step control, frame budget, CUDA graphs |
| `core.py` | Observation codec and running moments |
| `object_mode.py`, `object_migration.py` | Root-only object checks and checkpoint migration |

WM training is teacher-only, with **no `encoder_priv` module or latent bottleneck**.
The actor directly concatenates normalized command, policy and privileged
observations with the unnormalized object vector, plus the optional existing
height-CNN feature. Fixed object point
clouds and transformed point coordinates are excluded; actor inference no longer
runs the point-cloud transform, in either simulator collection or differentiable
imagined rollouts. Its hidden widths remain 512/256/256. The critic retains its
original inputs and learns detached lambda returns from the same imagined
trajectories. Student encoders, object estimators and student policies are neither
run during collection/evaluation nor trained. Their frozen modules are retained
only as frozen modules. Actor updates start after a short model warmup and
equation-parity check. Raw-teacher checkpoints carry an explicit input schema;
older latent-teacher checkpoints cannot resume this architecture. Start a new
run for this ablation; removing a nonlinear encoder cannot preserve its learned
policy exactly. Original PPO training is unchanged.
For the known skateboard observation layout, the actor also excludes the final
12 object-pose values from `object_`: those raw measurements already occur in
`priv`, where the existing normalization is retained. A parameterless prefix
view preserves the ten remaining object/contact features, leaving the original
22-dimensional `object_` available to the critic and world model. This projection
is enabled only when the observation configuration and widths prove that layout;
unknown layouts keep their complete object vector.

The optimizer is inspired by [SuperMimic](https://supermimic.github.io/) and extends
the physical state to the manipulated skateboard root. The
world model predicts generalized velocity changes at physics substeps, integrates
root poses and joint positions, and reconstructs body poses and velocities using
differentiable forward kinematics from the existing USD assets. Per-substep
contact forces/support and applied torques are supervised predictions. Contact
timers and force histories follow the original sensor equations. Actuator
delay/filter history and episode-constant randomized physics remain part of the
model state.

The default object model predicts the object's position,
quaternion, linear velocity and angular velocity (13 state components) by
integrating six predicted velocity increments. Wheel angles/velocities and unused
wheel torques are excluded from its network inputs, prediction heads and fitting
loss. The hidden layers and robot model keep their original capacity. Real
simulation wheels, wheel/domain randomization, teacher/critic inputs and all task
equations are unchanged. Foot/board forces and robot contact history remain
modeled because the existing reward and termination equations consume them.

Root mode requires the tracked object body to be its root; it rejects observations
or task terms that require articulated object state. Imagined wheel fields use
fixed placeholders solely to retain the shared context layout. This is a reduced
dynamics approximation: omitted wheel spin can affect real rolling and sliding,
so task-interface parity does not imply exact physical prediction. Validation in
root mode omits wheel errors and averages its normalized dynamics score over the
remaining terms; compare raw root/body errors between modes, not that aggregate.

```bash
python scripts/train.py algo=wm_train task=G1/vaic/skateboard_tea
```

To continue from an articulated WM checkpoint **with the same raw-teacher input
schema**, supply the usual
`checkpoint_path=/absolute/path/checkpoint_latest.pt`.
Compatible physical-v2 checkpoints are automatically converted to root mode.
This projects the retained model weights and normalization statistics,
keeps the raw teacher/critic and their optimizers, and resets only
the WM optimizer because its parameter shapes changed. All target parameters come
from the checkpoint; the conversion is not a mathematically equivalent resume.
A subsequent root-to-root resume restores all optimizer states normally.

The [root-mode pilot](artifacts/wm_verification/object_root_20261117.json) passed
the real 4096-environment task/reference oracles, implemented-gradient checks,
finish-execution parity and teacher-only checks. Its thirteen collections took
322.3 seconds of collection+learning (24.41 seconds per steady cycle). First-frame
evaluation increased from 87.02 to 89.16 steps (repeat 89.01), with zero full-motion
success. The control used twelve collections/~301 seconds and retained WM Adam,
whereas conversion resets it; this is not a causal isolation of representation
alone or proof of long-run convergence. This pilot used the previous, larger
7/.1 update budget. See the [comparison report](artifacts/wm_verification/object_root_comparison_20261117.json)
for settings, initial-condition matching, verification and limitations. CPU
regression testing passed 366 tests, with 13 environment-dependent tests skipped.

Optimization units default to SuperMimic-style Bryson nominal errors:
`root_pos=0.02` m, `root_rotation=0.05` rad, `joint_pos=0.1` rad and
`joint_velocity=3.0` rad/s (`root_velocity` keeps 2.5). Generic keys apply to every
asset. `object_root_pos`/`object_root_rotation`/`object_root_velocity` override
only the object root terms, so the board can be weighted separately from the
robot root. The default `object_root_velocity=1.0` (instead of 2.5) gives the
board twist 6.25x weight; pass `algo.wm_loss_scales.object_root_velocity=2.5`
for the previous equal weighting.
`object_joint_pos`/`object_joint_velocity` override only articulated object
joints, and robot substep joint positions follow `joint_pos`. Hydra merges dict
overrides key by key, for example `'++algo.wm_loss_scales={root_pos:0.05}'`; the
legacy units are `root_pos=0.5, root_rotation=0.5, joint_pos=0.25,
joint_velocity=2.5`. This changes the WM training objective only. Reported
physical errors and task rewards keep their original definitions.

The contact transition processes all four physics substeps in order. It advances
the captured float32 sensor clock and computes each elapsed interval by subtracting
the previous update timestamp. A body is in contact when its force norm exceeds
the sensor's configured threshold. Its air timer increases only while airborne,
and its contact timer increases only while in contact; the other timer resets to
zero. At touchdown or detachment, the previous timer plus that elapsed interval
becomes the corresponding last-air or last-contact duration. Force history is
rebuilt newest first. The existing first-contact test remains
`(current_contact_time > 0) & (current_contact_time < step_dt + 1e-8)`.
No independently predicted timer or softened contact switch replaces these
equations; derivatives through the discrete contact switch are zero.

Original actor/critic observations are reconstructed from this shared physical
state, including noisy sensor histories and the task's pre/post-command timing.
Imagined reference trajectories advance directly through the existing motion
dataset from the replay start's motion ID and time. They preserve the original
pre/post-command timing and continue beyond a stored factual failure. Pure
step noise is resampled; the combined step/episodic noise of `ref_contact_pos_b`
stays fixed because existing replay does not separate its episodic offset.
Thus the real environment is unchanged, but the imagined noise process is not
fully equivalent for that observation. Future measured physical states and
stored `done` flags are not supplied to policy imagination. Rewards,
hard termination counters and discounts use tensor implementations of the original
task equations, including their cached-state timing. These exact equations remain
the factual fitting/validation path. Startup checks compare forward kinematics with PhysX, and
a real-transition oracle checks every original observation group, reward group,
termination and discount before actor learning can start. It also checks contact
sensor transitions against the captured substep forces and sensor clocks.

Actor optimization backpropagates imagined returns through frozen world-model
weights and the current critic, using the teacher's deterministic mean action. It
uses no PPO ratio/clipping objective or entropy bonus. Simulator collection keeps
Gaussian exploration, with standard deviation scheduled from `0.3` to `0.1` over
20 million frames. After each actor update, the critic fits the detached imagined
GAE/lambda targets once: all valid imagined samples from that actor minibatch
contribute to a single critic optimizer step. Its original VAIC input groups
and output reward groups are preserved.
Policy imagination additionally uses an analytic **soft termination** surrogate.
Continuous tracking errors and contact distance become sigmoid violation
probabilities (relative softness 0.05). Imagined lost contact is judged by the FK
foot-to-target distance alone, because the learned contact-force head misses most
contact events at multi-step horizons; `algo.imagination_contact_force_termination=true`
also requires the predicted force, as the real env does. A separate distribution
over consecutive violation runs preserves each term's configured grace period,
seeded once from its logged counter. It uses at most H+1 probability cohorts per
term even for the 5000-step object grace period; it adds no learned head.
The per-term processes are combined with an independent-event OR approximation.
These are surrogate probabilities, not calibrated simulator failure probabilities.
Gradients remain zero when failure cannot occur within H given the logged counter
and grace period; the surrogate does not silently replace delayed failure with
immediate termination.

Actor reward/terminal-value weights and critic lambda-return continuations use
the same soft survival. Probabilities are never converted to boolean failures;
critic regression includes all imagined states up to genuine time/motion truncation,
with detached targets. True motion-end/time-limit transitions retain bootstrapping.
The actual simulator's termination, success statistics, rewards and replay masks
are unchanged. W&B reports `actor/soft_termination` and the survival-weighted mean
`actor/termination_probability` each iteration, without terminal progress prints.
The legacy target-critic module is retained for checkpoint compatibility,
not used by the new actor/critic objective.

The default `algo.schedule_mode=fixed` follows SuperMimic's alternation: per
collection, `algo.wm_updates=48` WM minibatches are spread evenly between
`algo.actor_updates=6` actor steps, and every actor step imagines from a fresh
uniform replay batch of `algo.actor_batch_size=4096` starts (accumulated in
microbatches of 512). No start state is reused across actor steps; reusing a
pool for several epochs is a PPO structure that pathwise gradients do not need,
and it lets the actor exploit model error on the same states. After each actor
step the critic fits that rollout's detached lambda-return targets
(`critic_epochs` x `critic_num_minibatches`).

Imagined actor/critic rewards use only functions of the predicted kinematics
(`algo.imagination_kinematic_reward=true`, as in SuperMimic): the contact-timer,
impact, slip and torque-limit terms, which read the learned contact/torque heads,
are dropped, and the contact kernel keeps only its FK position factor. WM
fitting, validation and the simulator keep the exact task reward. Critic
bootstrap values and lambda-return targets are clamped per reward group at
`max_step_reward / (1 - gamma)` (`algo.critic_value_ceiling=true`), SuperMimic's
`v_max`; `actor/bootstrap_clamped_value` reports the bootstrap return it removes.

The world-model network is invariant to horizontal translation and to rotation
about gravity by default (`algo.wm_frame=heading`, as in SuperMimic and NeRD).
The robot root enters as its height, body-frame gravity and body-frame twist.
The board enters as its position and x/y axes in the robot heading frame and as
its twist in its own heading frame, the frame along which it rolls. Predicted
root increments are expressed in each asset's heading frame at the start of the
step and rotated to the world. Substep integration, FK and every task equation
are unchanged. `algo.wm_frame=world` restores raw world-frame root inputs and
outputs; checkpoints trained before this option need it, and the loader names
the flag.

For every non-robot asset, the network also receives the position of every
robot body link in that asset's root frame (`algo.wm_object_body_features=true`,
39 x 3 values per object for the G1). These positions are exact FK quantities of
the input state, not predicted outputs, so they stay consistent with the
predicted poses. They tell the model which body is near which part of the
object. The features do not depend on the object's shape, so a box or a cart
uses the same features as the board. Like the heading inputs, they are invariant
to yaw and horizontal translation. The full object rotation is used, so a
tumbling box keeps its faces in its own frame. Checkpoints trained without these
features need `algo.wm_object_body_features=false`.

By default the world model does not predict contact (`algo.wm_contact_model=false`),
as in SuperMimic. Contact force and timer fields are not network inputs, the
contact presence/force heads and their losses are removed, and the next state
depends only on the generalized state, the actuator state and the domain
parameters. Contact force is a function of that state, and contact timers and
force histories are only sensor bookkeeping for rewards. Applied torques stay a
learned output because the teacher observes them. Imagined contact fields keep
their start-state values. This mode therefore requires the kinematic imagined
reward and position-only imagined lost contact, and it rejects either switch.
The factual reward loss leaves out reward groups whose exact terms read contact
force or timers (`feet` and `object_tracking` in this task). Those exact contact
terms are not evaluated at all, and the end-effector force is not rotated into
the board frame. Validation reports these groups as
`validation/h*/reward_contact_group_<i>_rmse` and leaves them out of
`reward_rmse`, `reward_max_group_rmse` and `reward_weighted_rmse`.
`validation/h*/reward_contact_free_rmse` pools only the contact-free groups in
both modes, so it is the reward metric to compare across the two modes. A lost-contact rule with a
nonpositive scalar `frc_thres` is position-only and never reads force. With a
positive `frc_thres`, the model's hard lost-contact counter reads the
start-state force and is diagnostic only; actor imagination uses the soft
position rule in either case. Checkpoints
trained with contact prediction load only with `algo.wm_contact_model=true`, and
the loader names the flag.

`task.command.reset_sampling=return` enables SuperMimic's return-based RSI: the
motion timeline is cut into 0.2 s bins, the measured discounted return per bin
drives the reset distribution (half uniform, the rest focused on low-return
bins), with options under `task.command.return_sampler`. It changes the
simulator task itself, so use the same setting for PPO baselines. The task-file
default `uniform` keeps the original uniform start phase; WM training uses
`return` through `algo.task_defaults` (see below).

The `algo.schedule_mode=epochs` alternative samples fixed replay-index pools once per
collection and reshuffles each pool for three epochs. Each epoch trains 16 WM
minibatches of 1024 eight-step windows, then eight actor minibatches of 512 starts.
The model therefore takes **48 optimizer steps**, the actor makes **24 optimizer
proposals**, and the critic takes **24 steps** per collection after warmup and
the equation oracle. KL rejection can reduce the accepted actor count. The critic
takes one step per actor minibatch, not another nested 3x8 loop. Every actor
minibatch generates a fresh differentiable rollout under the current actor and
model; only starting states are reused, never stale graphs or action trajectories.
WM fitting uses eight-step windows; policy
imagination independently uses `algo.imagination_horizon=16`. The shorter policy
rollout limits compounded dynamics error; it does not reduce network capacity,
physics substeps or the H8 factual model-training window. H32 remains available
with `algo.imagination_horizon=32`.
The actor start pool contains 4096 states, split into eight logical minibatches
of 512. Each minibatch has one optimizer step and one KL-control decision;
microbatch accumulation remains available for memory limits. The fixed replay
pools store CPU indices only, avoiding materialization of whole epochs of packed
states. Model minibatches retain bounded prefetch and fixed-shape CUDA graphs.
Critic inputs and targets are
detached, and its logical minibatches accumulate gradients in GPU microbatches
of 8192. Actor microbatches write this detached data into a reusable GPU cache
when sufficient memory is available; otherwise the same minibatches stream
from CPU storage.
After `algo.warmup_model_updates=10` model optimizer steps and the equation oracle,
epoch-mode actor learning can begin after the first 16 model steps.
Validation runs once after that first model epoch; `validation/model_updates_at_check`
identifies this point. Its errors do not describe the model after all 48 updates.
Prediction-accuracy metrics do not veto policy learning by default. A finite
objective/gradient and the one-time task-equation oracle are still required.
`wm/actor_horizon` records the horizon actually used;
`validation/approved_horizon` separately describes measured model accuracy.
This remains a VAIC adaptation: its raw privileged teacher inputs, object dynamics,
consecutive-failure survival surrogate and Adam/KL-controlled updates differ from the
local SuperMimic controller. These settings establish the intended learning
budget; they do not establish faster time to task performance.

Model fitting uses autoregressive supervision. Held-out environments evaluate
1-, 2-, 4-, 8-, 16- and 32-step predictions, capped by collection and imagination
length, against a constant-velocity baseline, with
absolute physical-error limits and checks for contact, reward and termination
errors. First-contact timing is checked on the sensor bodies used by the active
task reward equations (both ankles for `skateboard_tea`). Unused wrist timers do
not determine readiness; all sensor forces/auxiliaries are still checked, and
`all_sensor_contact_event_*` metrics retain their event diagnostics separately.
First-contact validation also requires at least four true contact events
at each checked horizon and at least 50% recall, aggregated over valid transitions;
event-free samples cannot hide missed contacts. These thresholds are configurable
with `algo.validation_min_contact_events` and
`algo.validation_max_contact_event_miss_rate`. These are diagnostic by default.
With the optional `algo.require_model_validation=true`, contact failures cap
actor rollouts at two steps. In that mode, two-step BPTT requires the equation oracle, warmup and three
consecutive successful checks of physical state, force/torque, pooled reward RMSE
and termination accuracy. It does not claim that discrete contact events are
accurate: hard contact switches have zero pathwise derivative. Longer four/eight
step rollouts additionally require contact fidelity at every preceding checked
horizon. Physical/reward/termination regressions close the gate again.
`validation/short_pathwise_ready` reports the repeated short-horizon checks;
`validation/contact_ready` reports contact checks in the current validation batch.
`wm/actor_updates` counts actual actor optimizer steps per collection.
`wm/policy_ready` indicates update admission, not a model-accuracy certificate.
`wm/accuracy_gate_enabled` distinguishes the default warmup admission from the
optional validation gate. A passing equation oracle alone
does not establish model accuracy or policy performance.

Every collection batch retains data from all simulator environments. A fixed
environment split separates model-training replay from held-out validation replay
(every tenth environment by default). Their default capacities are 524,288 and
131,072 transitions. Both use lossless CPU storage that shares adjacent states
and preserves exact terminal/reset exceptions. WM rollout buffers also reside on
the CPU; sampled training batches move to the GPU. CPU rollout storage places
each complete simulation step contiguously in memory while preserving the
logical `[environment, time]` axes, dtypes and all current/terminal states.
Replay packing retains explicit group copies: direct gathering into strided
output slices regressed full-scale throughput and was reverted. The rollout
layout has CPU data-parity coverage; further full-scale timing is recorded with
the runtime verification artifacts. Raw observations are captured
before VecNorm, and sequence masks prevent training across reset boundaries.
Model-fit transfers omit derived observation groups that its loss never reads,
and send adjacent context frames once plus exact successor exceptions. They are
expanded to the original packed shape on the GPU; factual physical/contact
targets, reward targets and precision are unchanged. Actor sampling transfers
only independent non-reset start states, rather than an entire factual sequence.
Validation and equation oracles retain complete observations.
Replay and rollout buffers add substantial host-memory requirements. The
[previous 4096-environment H32 pilot](artifacts/wm_verification/h32_default_pilot_20261117.json)
measured a 9.93 GB CPU rollout buffer and 47.00 GB peak process RSS. Its PyTorch
CUDA allocation peak was 5.51 GB; simulator/native/driver GPU allocations are
additional. Longer runs can have different memory peaks as replay fills.

The ordinary training command needs no additional arguments. Epoch-mode budgets use
`wm_epochs=3`, `wm_num_minibatches=16`, `ppo_epochs=3`, `num_minibatches=8`,
`critic_epochs=3`, `critic_num_minibatches=8`, and `algo.wm_batch_size=1024`.
The inherited `ppo_epochs` name controls BPTT epochs here, not a PPO objective.
Other settings include
`algo.wm_sequence_length` (8), `algo.imagination_horizon` (16),
`algo.actor_batch_size` (4096), `algo.actor_microbatch_size` (1024), and
`algo.cpu_threads` (4, CPU collection/replay tensor workers only).
Only factual model sequence length is limited by the collection length.
The epoch budgets above apply to `algo.schedule_mode=epochs`; legacy `frames`
also remains available. Unlike the old frame schedule, epoch mode uses fixed pools per collection, so changing the
number of simulator environments changes learning per collected transition.
Each collection contains 4096 x `train_every` (16) = 65,536 fresh simulator transitions; faster cycles
do not come from relabeling a smaller collection as one iteration. The previous 224-model /
12–13-actor budget remains available in legacy `algo.schedule_mode=frames`
with `algo.wm_predictions_per_frame=14`
and `algo.actor_starts_per_frame=0.4`. Compare evaluated performance against total
wall-clock time, not iteration count; fewer updates alone do not prove convergence.
The optional validation gate additionally waits for repeated accuracy checks.

For comparison, `ppo_vel_train` runs three epochs with eight minibatches, so its
teacher actor (including `encoder_priv`) and critic each take 24 optimizer steps
per collection. Each PPO minibatch contains 16,384 collected transition slots.
Its train phase also takes eight student-adaptation and eight student-policy
distillation steps; WM keeps those students frozen. A WM actor update differentiates
through a 16-step model rollout, so update counts do not measure equal work.
Each WM critic step accumulates its minibatch's valid imagined samples in
microbatches of up to 8192. The 24 critic steps use newly generated targets,
so this matches PPO's optimizer count, not PPO's training-data distribution.
Compared with the immediately preceding 16/1/1 budget, the new budget has three
times the model prediction slots and actor imagined trajectories. Encoder removal
does not by itself establish a shorter iteration or faster convergence.
In `skateboard_tea`, removing the 256-dimensional privileged latent and excluding
384 fixed point-cloud coordinates plus 12 duplicate object-pose values changes
actor input from 861 to 2329 features. The raw actor has 1,398,062 parameters
versus 1,255,726 for the former actor and privileged encoder together (11.3% more).
It removes the encoder bottleneck and
point-cloud transform while increasing the actor's first-layer work. The points
are loaded once from the motion file and copied unchanged across environments;
the existing contact-target coordinates retain the object's randomized scale.

The [point-cloud removal verification](artifacts/wm_verification/raw_teacher_geometry_removed_verification_20261117.json)
passed 414 CPU tests (13 environment-dependent skips) and two full
4096-environment GPU iterations from scratch. Every iteration executed 48 model,
24 accepted actor and 24 critic steps. Eager/captured actor gradients and payloads
matched, task/reference equations and finite-difference checks passed, students
remained unchanged, and a raw-teacher checkpoint was saved. With timing
synchronization enabled, the cold iteration took 23.83 seconds and the next
iteration took 17.59 seconds including collection. The latter is one warm sample.
This is technical readiness, not a
convergence result or a matched demonstration of speedup from encoder removal.
No full training run was launched.
The fixed geometry group remains in the shared environment and replay schema.
Inherited PPO construction uses its transform once to initialize compatibility
modules; the WM teacher never calls it during collection or training. Tests
forbid those runtime calls and verify that changed, missing, or invalid geometry
has no influence on the raw teacher's actions or gradients.

The [previous 32/1/8-budget verification](artifacts/wm_verification/fresh_iteration_defaults_20261117.json)
ran the actual 4096-environment task with that budget and automatic legacy
checkpoint conversion: 23 collections in 300.20 seconds of collection+learning,
with exactly 32 model / 1 actor / 8 critic updates in every collection. Steady
cycle time was 12.80 seconds, versus 24.41 seconds for the previous root-model
budget (47.5% less time). Each cycle still collected 131,072 real transitions.
CPU regression tests passed 375 cases with 13 environment-dependent skips;
the GPU reference/equation, implemented-gradient, reconstruction-parity and
teacher-only checks also passed. Students remained frozen and unchanged.

Speed improved, but the short policy evaluation did not: first-frame episode
length changed from 87.02 to 86.01 steps (repeat 85.90), and full-motion success
remained zero. More frequent data collection alone has not demonstrated faster
policy convergence. At the measured steady speed, 6100 cycles extrapolate to
21.7 hours before startup/evaluation/checkpoint overhead, not ten hours. No full
training run was launched by this verification. These timings and policy results
do not measure the current raw-teacher 48/24/24 budget. A running process retains its loaded
settings; the new defaults apply on its next launch.

Training writes `metrics.jsonl` in the W&B run
directory, including physical validation errors, gate status, model/actor/critic
losses and real episode statistics. `actor/action_delta_rms` and `actor/update_mean_kl`
measure accepted changes at replay starts; the optimization objective remains
the differentiated model return.
`actor/collection_mean_kl` and `actor/collection_action_delta_rms` measure the
combined effect of all actor updates between collections on a fixed replay
anchor, using one normalization snapshot and the initial exploration standard
deviation. They expose drift hidden by per-update KL; they are not exact KL
against each transition's collection policy. Sampling these anchors preserves
the training RNG. Set `algo.actor_drift_batch_size=0` to disable this diagnostic.
`algo.validation_horizon=32` requests predictions through H32, capped by collection
and actor horizon; the current H16 actor default therefore reports through H16.
Long-horizon diagnostics do not change the optional H2/H4/H8 admission gate or
the existing H8 critic diagnostic, and factual-action prediction accuracy does
not establish accuracy along the actor's imagined trajectories.

By default (`algo.actor_optimizer=adam`) each actor minibatch takes one
SuperMimic-style Adam step at a fixed `algo.actor_adam_lr=2e-4` with
`algo.actor_adam_beta2=0.95`, after the global norm clip and a per-element
`algo.actor_grad_clip_value=0.1` clip. There is no KL trust region, so the
decaying exploration std no longer shrinks the update; `actor/update_mean_kl` and
`actor/action_delta_rms` are still measured on the same starts as diagnostics.
An optional start filter, `algo.actor_start_violation_tolerance=N` (default -1,
off), excludes actor starts whose pose/object consecutive-violation counters
exceed N steps (an already falling robot) and reports the kept share as
`actor/healthy_start_fraction`. Lost-contact counters are never filtered: those
states are recoverable, and they are the only starts from which imagined contact
termination can fire within the horizon. Hidden actor layers use orthogonal gain
sqrt(2); only the action head starts at 0.01. The legacy trust region below is
selected with `algo.actor_optimizer=kl`.

In `kl` mode, WM actor updates start at `algo.actor_lr=1.5e-5`, with
the legacy `algo.encoder_lr=5e-6` used only for the optional teacher height CNN.
The critic retains `algo.lr=3e-4`. A measured Gaussian KL on valid real replay
starts controls each actor update: target `algo.actor_target_kl=0.01`, maximum
`algo.actor_max_kl=0.02`. Oversized Adam proposals are interpolated and checked
again using teacher forwards, without repeating BPTT. A completely rejected
proposal restores weights and Adam moments; accepted steps adapt the next rate
up to `algo.actor_lr_max=1e-4`, maintaining the height/actor rate ratio when present.
These are calibrated starting settings, not proven optimal learning rates or a
guarantee of improved returns. The KL bound applies to that update's sampled
starts. `wm/actor_updates` counts accepted updates and
`wm/actor_update_attempts` counts attempted BPTT updates. Physical-WM checkpoints
with the same raw-teacher input schema retain their learned weights/moments. Checkpoints lacking the actor controller's
state start it at the configured initial rates; checkpoints containing that state
also restore its adaptive rates.

`performance/wm_{replay,sample,model,validation}_seconds` separates those learner
costs. `performance/wm_actor_seconds` includes both actor imagination/update and
critic fitting; its component timers are `wm_imagination_seconds` and
`wm_critic_seconds`, so do not add the combined timer to its components.
Set `algo.profile_timing=true` for synchronized CUDA phase timings.
WM reward, tracking, episode length and success statistics are logged to W&B
every iteration when completed episodes are available. Update counts, losses,
timings and the explicit iteration number are also recorded every iteration.
There is no separate terminal progress report.
WM also refreshes `checkpoint_latest.pt` between iterations after approximately
`algo.checkpoint_interval_seconds=600` seconds, in addition to the existing
numbered checkpoints. Set this interval to zero to disable timed saves. The
latest checkpoint contains the teacher, WM, critic, optimizers, normalization
and frame-schedule state; CPU replay and exact simulator/RNG state are rebuilt
on resume. `checkpoint/latest_iteration` is recorded in W&B when refreshed.
Sampling returns owned tensors directly, without concatenating/reordering/cloning
full batches repeatedly. Forward kinematics batches independent sibling links.
These optimizations retain model capacity, 1024 sequences per model batch and
eight-step autoregressive supervision. Frame-based scheduling now determines
the number of model updates.

On CUDA, `algo.cuda_graph_wm=true` caches the same fixed-shape WM loss and
parameter-gradient computation. Optimizer steps, finite checks and clipping
remain outside the graph; the first update includes a one-time capture cost.
`algo.replay_prefetch=true` prepares the next unchanged sample on a worker and
transfers pinned sample buffers on a separate stream, overlapping this work
with GPU fitting. Whole replay storage remains ordinary CPU memory. Both paths
can be disabled independently for diagnosis. Model dimensions, precision,
training samples, update counts, autoregressive length and task equations are
unchanged. Capture and prefetch require additional host/GPU buffer memory.
With prefetch enabled, sample timing measures exposed wait time, so end-to-end
iteration time is the appropriate speed comparison.

`algo.cuda_graph_actor=true` also captures teacher/world-model/critic forwards at
the configured imagination horizon
and the actor backward. Microbatch gradients accumulate before the single
logical optimizer step; clipping and the KL controller remain outside capture.
Graph execution reads updated model weights and in-place normalization snapshots.
A full-task B256/H32 parity check of actor loss, every actor gradient and the
detached critic targets against eager execution matched exactly. No model
dimensions or precision were reduced.
First-iteration timings include capture and should be distinguished from replay.
When the detached critic cache fits on GPU, actor microbatches now write directly
into its owned slices before captured outputs are reused. This avoids the previous
GPU-to-CPU staging, CPU concatenation and CPU-to-GPU copy of the full critic batch.
The valid-mask shuffle still uses the original CPU RNG; insufficient GPU memory
retains the CPU staging path with the same minibatches and targets.
If a new cache fails its memory-reserve check, the allocator releases unused
cached memory once per payload shape and rechecks actual driver-free memory.
Live tensors and captured graph storage remain owned, and the 4 GiB reserve is
unchanged. A real shortage still uses CPU staging; replay storage stays on CPU.

Iteration counts are not directly comparable to local SuperMimic. One VAIC
collection contains 32 simulator steps per environment; SuperMimic collects one
per raw runner step and logs every ten. Its actor processes the logical batch of
4096 together, whereas this adaptation uses eight microbatches of 512.
Matching sampled-transition ratios therefore does not match execution cost.
Motion-dataset size changes the sampled reference content, not these fixed batch
sizes.

The [VAIC budget comparison](artifacts/wm_verification/vaic_budget_report_20261117.json)
uses the same earlier checkpoint, startup DR seed and repeated first-frame
evaluation. Initial mean episode length was 89.16 and return was 12.44. The latest
stopped user run had no checkpoint, so it could only be analyzed from logs.

| WM / actor budget, horizon | Actual collection + learning | Steady seconds / collection | Final episode steps | Final return |
| --- | ---: | ---: | ---: | ---: |
| 14 / .4, H32 (previous) | 376.6 s | 121.5 | 90.02 | 12.68 |
| 7 / .1, H32 | 317.4 s | 51.4 | 91.10 | 12.99 |
| 7 / .1, H16 (previous) | 314.1 s | 34.3 | 90.95 | 13.01 |

Steady timing excludes the first collection's capture cost. Budgets stop between
whole iterations, so time and frame counts are not identical: the three runs
collected 393,216 / 786,432 / 1,179,648 new frames. Fewer repeated actor updates
reduced mean collection KL from .576 to .081–.083. This is a lower optimization
budget and shorter rollout, not an identical-work speedup. H16 and H32 achieved
comparable short-run evaluation; H16 is selected for shorter iterations and
more frequent fresh data, not demonstrated better wall-time convergence.
The H32 reduced-budget run used CPU critic staging, whereas the other two used
GPU caches; cache placement also contributes to their timing difference.
H16 bootstrap contributes about 41% of the actor objective versus 12% at H32.
Forward model errors remain material, and validation is capped at H16 under the
new default. All three full-motion success rates were zero. One seed/checkpoint
and evaluation repeats do not establish long-run convergence or superiority to
PPO. The model capacity, teacher/privileged encoder, CPU replay, GPU training and
original task definitions are unchanged.
The final regression suite passed 303 tests. A two-iteration `train.py` smoke run
with 64 environments and a small fixed diagnostic update budget verified teacher
gradients, per-iteration training metrics, atomic latest-checkpoint replacement
and checkpoint readability, without the removed WM terminal progress messages.
Its W&B mode was disabled; this verifies the logging path, not server delivery.

The earlier [runtime and logging follow-up](artifacts/wm_verification/runtime_logging_report_20261117.json)
records the stopped run's 116.6-second median learner time and the subsequent
transport/microbatch experiments. A B512 test reduced actor computation but
triggered CPU critic fallback and left only 0.82 GB driver-free memory at the
recorded cache check. That experiment retained the then-default microbatch of
256. Those measurements use the previous learning budgets;
they do not establish time to a successful policy.

A later H16/7/.1 execution check used four CPU tensor workers, actor microbatches
of 512 and shared decoded context for observation/reward reconstruction. The
[12-collection control](artifacts/wm_verification/pose_control_20261117.json)
measured 24.69 seconds per steady collection+learning cycle, versus roughly
34–35 seconds in the user's preceding stopped run. Its workload remained
4096 environments × 32 real steps, 112 model updates and 3/4 actor updates per
collection. Replay stayed on CPU and all training stayed on GPU. Actual task
output/gradient parity passed for the reconstruction execution change. This is
about 41.8 hours for 6100 such cycles by linear extrapolation, excluding startup,
evaluation and checkpoint overhead; the requested ten-hour target is not reached.

An offline physical-loss diagnostic found wheel angle/velocity terms dominated
the total gradient direction on one sampled batch. Separating optimization units
improved held-out root/body prediction, but the subsequent
[300-second policy pilot](artifacts/wm_verification/pose_balanced_20261117.json)
did not improve first-frame evaluation length: 87.02 before, 85.49 after
(repeat 85.36), versus 86.45 after the control (repeat 86.36). All full-motion
success rates remained zero. These pilots include twelve collections each and
do not establish long-run convergence. Loss-scale overrides therefore remain
opt-in; lower model loss alone is not evidence of a better policy.

The [2026-11-17 verification report](artifacts/wm_verification/supermimic_alignment_report_20261117.json)
records 268 passing regression tests, the actual training entry point, and both
new and legacy physical-checkpoint resumes. A three-collection run at the previous
4096-environment, H32, 14/.4 defaults completed 393,216 real transitions, 672 model updates
and 38 accepted actor updates. The teacher and `encoder_priv` changed; student
parameters stayed unchanged. All accepted updates had measured mean KL below
0.02. Independent reference/equation checks and implemented-gradient checks
passed; these do not certify learned H32 dynamics accuracy.

At a matched diagnostic budget of 32 model updates and one B4096/H32 actor update,
the second collection's actor-plus-critic time fell from 37.66 to 8.57 seconds,
and collection-plus-training time from 51.48 to 22.79 seconds (55.7% less).
This measurement predates the final persistent-critic-cache fix; reuse of that
cache was then verified throughout that pilot. The previous larger 14/.4
budget took 126.9–131.4 seconds per collection, including simulation and training.
It is not faster per collection than the former small learning budget.

In that pilot's same-seed, first-frame evaluation, mean return changed
from 12.073 to 12.437 and episode length from 85.15 to 89.15 steps. Full-motion
success remained zero. Normalization was frozen within each evaluation but
learned during training, and native reset/cache residuals differed between
evaluations. One repeated evaluation measured return variation of 0.028 before
and 0.003 after training; this is not a confidence interval or evidence of faster
convergence than PPO. The full 800-million-frame run has not been performed.

Earlier runtime measurements in the [previous-budget report](artifacts/wm_verification/update_runtime_report_20261117.json)
used H8, 32 model updates and two small actor updates per collection. Their
32.38 → 15.53 seconds/collection result and 217-test result describe that older
configuration, not the present epoch-budget defaults. The
[original PPO timing](artifacts/wm_verification/ppo_runtime_4096_20261117.json)
was 3.15 seconds/collection on this machine. Neither comparison establishes
which algorithm reaches a given task performance first.

Contact-force predictions and their gradients remain approximations; exact task
equations do not make the learned dynamics an exact differentiable simulator.
Long-run convergence and performance relative to PPO require a matched training
comparison. Execution readiness and equation/gradient parity alone do not
establish those results.

Version-2 WM checkpoints include the physical model, actor/critic, optimizers,
normalization and context schema. Resume with
`checkpoint_path=/absolute/path/checkpoint_final.pt`; replay is rebuilt and
warmup/validation gates restart. Load with `algo=wm_train`: PPO and version-1
observation-space WM checkpoints are rejected. Physical/context schema changes,
including older independently predicted contact timers, also require a fresh run.
Teacher evaluation uses:

```bash
python scripts/play.py algo=wm_train task=G1/vaic/skateboard_tea \
  checkpoint_path=/absolute/path/checkpoint_final.pt
```

CPU regression checks: `python -m pytest tests -q`.

## Acknowledgments

This repository is built on top of [HDMI: Learning Interactive Humanoid Whole-Body Control from Human Videos](https://github.com/LeCAR-Lab/HDMI). We thank the authors for open-sourcing their work.

## Citation

If you find our work useful for your research, please consider citing us:

```bibtex
@article{li2026vaic,
  title = {VAIC: Vision-Guided Humanoid Agile Object Interaction Control via Decoupled Commands},
  author = {Li, Dongting and Wu, Qianyang and Chen, Xingyu and Li, Liang and Lin, Yuhang and Wu, Sikai and Zhang, Guoyao and Zhou, Mingliang and Xiang, Diyun and Zhang, Qiang and Xu, Renjing and Ma, Jianzhu},
  journal = {arXiv preprint arXiv:2606.09286},
  year = {2026}
}
```
