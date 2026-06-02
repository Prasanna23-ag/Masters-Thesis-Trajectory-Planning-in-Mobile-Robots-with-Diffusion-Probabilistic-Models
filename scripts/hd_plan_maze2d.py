import json
from turtle import pos
import numpy as np
from os.path import join
import pdb

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
from diffuser.utils.rendering import Maze2dRenderer
import os

class HLParser(utils.Parser):
    dataset: str = "birrt-dataset-v0"
    config: str = "config.maze2d_hl"

hl_args = HLParser().parse_args("plan")

class LLParser(utils.Parser):
    dataset: str = "birrt-dataset-v0"
    config: str = "config.maze2d_ll"


ll_args = LLParser().parse_args("plan")
# ################################# setup ######################################

# ---------------------------------- loading ----------------------------------#


n_samples = 500

loadpath = (hl_args.logbase, hl_args.dataset, hl_args.diffusion_loadpath)

hl_diffusion_experiment = utils.load_diffusion(
    hl_args.logbase,
    hl_args.dataset,
    hl_args.diffusion_loadpath,
    epoch=hl_args.diffusion_epoch,
)
hl_diffusion = hl_diffusion_experiment.ema
dataset = hl_diffusion_experiment.dataset
hl_policy = Policy(hl_diffusion, dataset.normalizer)

ll_diffusion_experiment = utils.load_diffusion(
    ll_args.logbase,
    ll_args.dataset,
    ll_args.diffusion_loadpath,
    epoch=ll_args.diffusion_epoch,
)
ll_diffusion = ll_diffusion_experiment.ema
ll_policy = Policy(ll_diffusion, dataset.normalizer)

env_eval = datasets.load_environment(hl_args.dataset)
renderer = Maze2dRenderer(env_eval)

target = env_eval._target
hl_cond = {
    hl_diffusion.horizon - 1: np.array([*target, 0, 0]),
}

total_rewards = []
scores = []
rollouts = []
plans = []
track_action = []


for i in range(n_samples):
    observation = env_eval.reset()
    init_obs = observation.copy()
    observation = env_eval._get_obs()
    rollout = [observation.copy()]

    hl_cond[0] = observation
    action, samples = hl_policy(hl_cond, batch_size=hl_args.batch_size)
    hl_plan = samples.observations
    hl_xy = hl_plan[0][:, :2] 
    renderer.composite(join(hl_args.savepath, f"hl_plan_{i}.png"),[hl_xy],ncol=1)
    
    hl_subgoals = hl_plan[0][:, :2]         ## saving subgoals for visualization

    B, M = hl_plan.shape[:2]
    ll_cond_ = np.stack([hl_plan[:, :-1], hl_plan[:, 1:]], axis=2)
    ll_cond_ = ll_cond_.reshape(B * (M - 1), 2, -1)
    ll_cond = {
        0: ll_cond_[:, 0],
        ll_args.horizon - 1: ll_cond_[:, -1],
    }

    _, ll_samples = ll_policy(ll_cond, batch_size=-1)
    ll_actions = ll_samples.actions
    ll_samples = ll_samples.observations
    ll_samples = ll_samples.reshape(B, (M - 1), ll_args.horizon, -1)
    ll_xy = ll_samples[0][:, :, :2].reshape(-1, 2) 
    renderer.composite(join(hl_args.savepath, f"ll_plan_{i}.png"),[ll_xy],ncol=1)

    ll_samples = np.concatenate(
        [
            ll_samples[:, 0, :1],
            ll_samples[:, :, 1:].reshape(B, (M - 1) * hl_args.jump, -1),
        ],
        axis=1,
    )

    # ---- FLATTEN LL ACTIONS INTO A SINGLE TIME SERIES ----
    la = np.array(ll_actions)

    # If shape is (B, S, H, 2), collapse batch
    if la.ndim == 4:
        la = la.reshape(-1, la.shape[2], la.shape[3])   # (S_total, H, 2)

    S, H, A = la.shape

    # Flatten to (T, 2)
    first = la[:, 0, :]                 # (S, 2)
    rest  = la[:, 1:, :].reshape(-1, A) # (S*(H-1), 2)

    actions_flat = np.concatenate([first, rest], axis=0)
    ll_sequence = ll_samples[0] 
    total_reward = []
    action_list = []

    max_episode_steps = env_eval.max_episode_steps
    finished = False
    t = 0
    while t < max_episode_steps:
        if finished:
            break
        else:
            if t < len(ll_sequence) - 1:
                next_waypoint = ll_sequence[t]
            else:
                next_waypoint = ll_sequence[-1].copy()
                next_waypoint[2:] = 0
        
            state = observation.copy()
            # --- USE LL DIFFUSER ACTIONS ---
            if t < len(actions_flat):
                action = actions_flat[t].astype(np.float32)
            else:
                action = actions_flat[-1].astype(np.float32)   # repeat last action

            action = np.clip(action, env_eval.action_space.low, env_eval.action_space.high)

            # --- USE LPID-CONTROL ACTIONS ---
            # action = next_waypoint[:2] - state[:2]  
            next_observation, reward, terminal, _ = env_eval.step(action)

            hl_xy = hl_plan[0][:, :2]
            ll_xy = ll_samples[0][:,:2]
            combined = np.concatenate([hl_xy, ll_xy], axis=0)
            

            t += 1
            total_reward.append(reward)
            score = env_eval.get_normalized_score(sum(total_reward))    

            ## update rollout observations
            rollout.append(next_observation.copy())
            if terminal or t >= max_episode_steps:
                finished = True
                print(
                    f" {i} / {n_samples}\t t: {t} | r: {reward:.2f} |  R: {sum(total_reward):.2f} | score: {score:.4f} | "
                )
                break
            observation = next_observation
    renderer.composite(join(hl_args.savepath, f"combined_plan_{i}.png"),[combined],ncol=1)
    rollouts.append(rollout)
    total_rewards.append(total_reward)
    scores.append(env_eval.get_normalized_score(sum(total_reward)))

    ## save result as a json file
    json_path = join(hl_args.savepath, f"idx{i}_rollout.json")
    json_data = {
        "score": score,
        "step": t,
        "return": total_reward,
        "term": terminal,
        "trajectory": [obs.tolist() for obs in rollout],
        "hl_subgoals": hl_subgoals.tolist()
    }
    json.dump(json_data, open(json_path, "w"), indent=2, sort_keys=True)


######################### Summary Metrics #########################

success_count = 0
collision_count = 0
episode_lengths = []

for i in range(n_samples):
    traj = rollouts[i]
    final_obs = traj[-1]

    # Maze2D success: distance to target < threshold
    dist_to_goal = np.linalg.norm(final_obs[:2] - target)
    if dist_to_goal < 0.5:
        success_count += 1

    # collision detection: negative reward in episode
    if any(r < 0 for r in total_rewards[i]):
        collision_count += 1

    episode_lengths.append(len(traj))

avg_score = np.mean(scores)
avg_steps = np.mean(episode_lengths)

print("\n==================== RESULTS ====================")
print(f"success: {success_count}/{n_samples}")
print(f"collisions: {collision_count}/{n_samples}")
print(f"avg score: {avg_score:.3f}")
print(f"avg steps: {avg_steps:.1f}")
print("=================================================\n")
