import torch
# import warp
import hydra
import numpy as np
import einops
import wandb
import logging
import os
import sys
import time
import datetime
import json

# Direct script execution otherwise can import another editable VAIC checkout.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf, DictConfig
from hydra.core.hydra_config import HydraConfig
from collections import OrderedDict
from tqdm import tqdm
from setproctitle import setproctitle

import active_adaptation as aa
from active_adaptation.utils.checkpoint import atomic_torch_save
from active_adaptation.utils.rollout import allocate_rollout_buffer
from isaaclab.app import AppLauncher
# from active_adaptation.utils.torchrl import SyncDataCollector
from torchrl.envs.utils import set_exploration_type, ExplorationType
from tensordict.nn import TensorDictModuleBase

# local import
from helpers import make_env_policy, EpisodeStats, evaluate, apply_algo_task_defaults

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(FILE_PATH, "..", "cfg")

@hydra.main(config_path=CONFIG_PATH, config_name="train", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    # Training-only task settings of the selected algorithm (WM); explicit
    # task.* command-line overrides win. Applied before the cfg is saved.
    applied, skipped = apply_algo_task_defaults(cfg, HydraConfig.get().overrides.task)
    if applied or skipped:
        print(f"algo task defaults: applied {applied}; skipped (absent from task) {skipped}")
    
    print(f"is_distributed: {aa.is_distributed()}, local_rank: {aa.get_local_rank()}/{aa.get_world_size()}")
    app_launcher = AppLauncher(
        OmegaConf.to_container(cfg.app),
        distributed=aa.is_distributed(),
        device=f"cuda:{aa.get_local_rank()}"
    )
    simulation_app = app_launcher.app

    run = wandb.init(
        job_type=cfg.wandb.job_type,
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        tags=cfg.wandb.tags,
    )
    run.config.update(OmegaConf.to_container(cfg))
    os.makedirs(run.dir, exist_ok=True)  # disabled W&B runs may not create it
    
    default_run_name = f"{cfg.exp_name}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    run_idx = run.name.split("-")[-1]
    run.name = f"{run_idx}-{default_run_name}"
    setproctitle(run.name)

    cfg_save_path = os.path.join(run.dir, "cfg.yaml")
    OmegaConf.save(cfg, cfg_save_path)
    run.save(cfg_save_path, policy="now")
    run.save(os.path.join(run.dir, "config.yaml"), policy="now")
    metrics_path = os.path.join(run.dir, "metrics.jsonl")

    env, policy, vecnorm = make_env_policy(cfg)

    import inspect
    import shutil
    source_path = inspect.getfile(policy.__class__)
    target_path = os.path.join(run.dir, source_path.split("/")[-1])
    shutil.copy(source_path, target_path)
    wandb.save(target_path, policy="now")

    frames_per_batch = env.num_envs * cfg.algo.train_every
    total_frames = cfg.get("total_frames", -1) // aa.get_world_size()
    total_frames = total_frames // frames_per_batch * frames_per_batch
    total_iters = total_frames // frames_per_batch
    save_interval = cfg.get("save_interval", -1)
    checkpoint_interval_seconds = cfg.algo.get("checkpoint_interval_seconds", 0.)
    last_checkpoint_time = time.perf_counter()

    # WM iterations contain substantially more learner work. Report completed
    # real episodes every collection instead of hiding them for ~32 iterations.
    # Preserve the original PPO reporting cadence.
    log_interval = 1 if getattr(policy, "log_every_iteration", False) else (env.max_episode_length // cfg.algo.train_every) + 1
    logging.info(f"Log interval: {log_interval} steps")

    stats_keys = [
        k for k in env.reward_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device="cpu" if getattr(policy, "requires_next_observations", False) else env.device)

    def save(policy, checkpoint_name: str, artifact: bool=False):
        nonlocal last_checkpoint_time
        ckpt_path = os.path.join(run.dir, f"{checkpoint_name}.pt")
        state_dict = OrderedDict()
        state_dict["wandb"] = {"name": run.name, "id": run.id}
        state_dict["policy"] = policy.state_dict()
        state_dict["env"] = env.state_dict()
        state_dict["cfg"] = cfg
        if vecnorm is not None:
            state_dict["vecnorm"] = vecnorm.state_dict()
        if checkpoint_name == "checkpoint_latest":
            atomic_torch_save(state_dict, ckpt_path)
        else:
            torch.save(state_dict, ckpt_path)
        if artifact:
            artifact = wandb.Artifact(
                f"{type(env).__name__}-{type(policy).__name__}",
                type="model"
            )
            artifact.add_file(ckpt_path)
            run.log_artifact(artifact)
        run.save(ckpt_path, policy="now", base_path=run.dir)
        last_checkpoint_time = time.perf_counter()
        if not getattr(policy, "requires_next_observations", False):
            logging.info(f"Saved checkpoint to {str(ckpt_path)}")

    assert env.training
    def should_save(i):
        if not aa.is_main_process():
            return False
        return i > 0 and save_interval > 0 and i % save_interval == 0

    # 4. --- Training Loop ---
    carry = env.reset()
    rollout_policy: TensorDictModuleBase = policy.get_rollout_policy("train")
    needs_next_obs = getattr(policy, "requires_next_observations", False)
    next_keys = ["done", "terminated", "discount", "reward", "stats", "is_init", "adapt_hx"]
    if needs_next_obs:
        next_keys += ["truncated", "wm_observation"]

    with torch.inference_mode():
        tmp_carry = rollout_policy(carry.clone(False))
        tmp_td, probe_carry = env.step_and_maybe_reset(tmp_carry.clone(False))
        tmp_td["next"] = tmp_td["next"].select(*next_keys, strict=False)
        if needs_next_obs:
            # The probe advanced physics. Do not pair stale reset observations
            # with a state two simulator transitions later.
            carry = probe_carry

    T = cfg.algo.train_every
    device = env.device

    # Physical WM context/reference snapshots are training-only and much larger
    # than policy inputs. Stream them to host memory instead of reserving most
    # GPU memory for duplicate current/next rollout states.
    buffer_device = "cpu" if needs_next_obs else device
    data_buf = allocate_rollout_buffer(
        tmp_td, T, device=buffer_device, contiguous_steps=needs_next_obs,
    )
    logging.info(f"Data buffer size: {data_buf.bytes() / 1e6:.2f} MB")

    if aa.is_main_process():
        progress = tqdm(range(total_iters))
    else:
        progress = range(total_iters)

    env_frames = 0
    start_iter = env.current_iter
    for i in progress:
        rollout_start = time.perf_counter()
        with torch.inference_mode(), set_exploration_type(ExplorationType.RANDOM):
            torch.compiler.cudagraph_mark_step_begin() # for compiled policy
            env.set_progress(start_iter + i)
            for step in range(cfg.algo.train_every):
                carry = rollout_policy(carry)
                td, carry = env.step_and_maybe_reset(carry)
                td["next"] = td["next"].select(*next_keys, strict=False)
                data_buf[:, step] = td
            if not needs_next_obs:
                policy.critic(data_buf)
                values = data_buf["state_value"]
                data_buf["next", "state_value"] = torch.where(
                    data_buf["next", "done"],
                    values, # a walkaround to avoid storing the next states
                    torch.cat([values[:, 1:], policy.critic(carry.copy())["state_value"].unsqueeze(1)], dim=1)
                )
        rollout_time = time.perf_counter() - rollout_start

        episode_stats.add(data_buf)
        env_frames += data_buf.numel()

        info = {}
        if i % log_interval == 0 and len(episode_stats):
            for k, v in sorted(episode_stats.pop().items(True, True)):
                key = "train/" + ("/".join(k) if isinstance(k, tuple) else k)
                info[key] = torch.mean(v.float()).item()
        training_start = time.perf_counter()
        info.update(policy.train_op(data_buf))
        training_time = time.perf_counter() - training_start
        info.update(env.extra)
        info.update(env.stats_ema)

        if hasattr(policy, "step_schedule"):
            policy.step_schedule(i / total_iters)

        info["env_frames"] = env_frames
        info["iteration"] = start_iter + i
        info["rollout_fps"] = data_buf.numel() / rollout_time
        info["training_time"] = training_time

        if should_save(i):
            save(policy, f"checkpoint_{i}")
        elif (aa.is_main_process() and checkpoint_interval_seconds > 0
              and time.perf_counter() - last_checkpoint_time >= checkpoint_interval_seconds):
            save(policy, "checkpoint_latest")
            info["checkpoint/latest_iteration"] = start_iter + i

        if aa.is_main_process():
            # print(OmegaConf.to_yaml({k: v for k, v in info.items() if (isinstance(v, (float, int)) and not k.startswith("performance_reward"))}))
            run.log(info)
            # Keep diagnostics available even for offline/disabled W&B runs.
            with open(metrics_path, "a", encoding="utf-8") as metrics_file:
                record = {key: value.detach().item() if isinstance(value, torch.Tensor) and value.numel() == 1 else value
                          for key, value in info.items()}
                record["iteration"] = start_iter + i
                metrics_file.write(json.dumps(record) + "\n")

    # 5. --- Finalization and Cleanup ---
    if aa.is_main_process():
        save(policy, "checkpoint_final", artifact=True)

    policy_eval = policy.get_rollout_policy("eval")
    info, trajs, stats, policy_trajs = evaluate(env, policy_eval, render=cfg.eval_render, seed=cfg.seed)
    run.log(info)
    with open(metrics_path, "a", encoding="utf-8") as metrics_file:
        metrics_file.write(json.dumps({"evaluation": info, "env_frames": env_frames}) + "\n")

    wandb.finish()
    os._exit(0)
    env.close()
    simulation_app.close()

    run_id = run.id
    project = run.project
    entity = run.entity
    run_path = f"{entity}/{project}/{run_id}"
    
    return run_path



if __name__ == "__main__":
    main()
