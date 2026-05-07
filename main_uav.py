"""
Entry point for UAV Semantic Communication experiments with COLA.

Usage:
    python main_uav.py --seed 1 --cuda
    python main_uav.py --seed 1 --num_devices 5 --regular_alpha 0.5
"""

import os
import numpy as np
# NumPy 2.0 compat: gym 0.25 references np.bool8 which was removed
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import torch
import wandb

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
cpu_num = 1
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)

import argparse

# Python < 3.9 (e.g. conda py3.8): BooleanOptionalAction was added in 3.9
if not hasattr(argparse, "BooleanOptionalAction"):

    class BooleanOptionalAction(argparse.Action):
        def __init__(
            self,
            option_strings,
            dest,
            default=False,
            type=None,
            choices=None,
            required=False,
            help=None,
            metavar=None,
        ):
            if type is not None or choices is not None:
                raise ValueError("BooleanOptionalAction does not support type or choices")
            new_os = []
            for s in option_strings:
                new_os.append(s)
                if s.startswith("--"):
                    new_os.append("--no-" + s[2:])
            super().__init__(
                option_strings=new_os,
                dest=dest,
                nargs=0,
                default=default,
                required=required,
                help=help,
                metavar=metavar,
            )

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, not option_string.startswith("--no-"))

    argparse.BooleanOptionalAction = BooleanOptionalAction

import gym
import numpy as np

from environments import *
from agent import SacAgent


def run():
    parser = argparse.ArgumentParser(
        description="COLA-SemCom: Multi-Objective UAV Semantic Communication"
    )

    # --- environment ---
    parser.add_argument("--env_id", type=str, default="UAV-SemCom-v0")
    parser.add_argument("--num_devices", type=int, default=5)
    parser.add_argument(
        "--num_uavs", type=int, default=1,
        help="Number of UAVs. >=2 routes to UAV-SemCom-Multi-v0 + MultiAgentSemMORL.",
    )
    parser.add_argument("--area_size", type=float, default=500.0)
    parser.add_argument("--uav_height", type=float, default=100.0)
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument(
        "--device_mobility",
        type=str,
        default="none",
        choices=["none", "line", "drift"],
        help="IoT ground devices: none (fixed) | line (slow constant velocity, edge reflect) | drift (small random walk)",
    )
    parser.add_argument(
        "--device_speed",
        type=float,
        default=0.0,
        help="Device speed (m/s) when mobility is on; if 0 and mobility is not none, env uses 0.5 m/s",
    )

    # --- COLA core ---
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--cuda_device", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=2000000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent_coef", type=float, default=0.2,
                        help="SAC entropy coefficient. Used as fixed alpha when "
                             "--no-entropy_tuning, else as the initial alpha.")
    parser.add_argument(
        "--entropy_tuning", action=argparse.BooleanOptionalAction, default=True,
        help="Auto-tune SAC entropy alpha to target -|A|. Disable with "
             "--no-entropy_tuning to use a fixed --ent_coef (lower alpha "
             "stabilises large-M training where target -|A| over-explores).")
    parser.add_argument("--eval_interval", type=int, default=20000)

    # --- latent encoder ---
    parser.add_argument("--latent_dim", type=int, default=50)
    parser.add_argument("--use_avg", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--encoder_update_freq", type=int, default=1)
    parser.add_argument("--use_encoder_hardupdate", action=argparse.BooleanOptionalAction, default=False)

    # --- policy / critic architecture ---
    parser.add_argument("--Policy_use_latent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--Policy_use_s", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--Policy_use_w", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--Policy_use_target", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--Critic_use_both", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--Critic_use_s", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--Critic_use_a", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--Use_Policy_Preference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--Use_Critic_Preference", action=argparse.BooleanOptionalAction, default=True)

    # --- COR (Conflict Objective Regularization) ---
    parser.add_argument("--regular_alpha", type=float, default=0.5,
                        help="COR regularization strength")
    parser.add_argument("--regular_bar", type=float, default=0.25,
                        help="COR stiffness threshold")
    parser.add_argument("--consider_other", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--old_Q_update_freq", type=int, default=1)

    # --- misc ---
    parser.add_argument("--prefer", type=int, default=0)
    parser.add_argument("--buf_num", type=int, default=0)
    parser.add_argument("--q_freq", type=int, default=1000)
    parser.add_argument("--train_with_fixed_preference", action="store_true",
                        default=True)
    parser.add_argument("--Use_pc_grad", action="store_true", default=False)
    parser.add_argument("--step_random", action="store_true", default=False)
    parser.add_argument("--EA_policy_num", type=int, default=0)
    parser.add_argument("--RL_policy_num", type=int, default=0)
    parser.add_argument("--warm_steps", type=int, default=8000000)
    parser.add_argument("--fixed_weight", type=float, nargs='+', default=None,
                        help="Fixed preference weight for Weighted Sum SAC")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name for log directory")
    parser.add_argument("--model_saved_step", type=int, default=100000,
                        help="Save model checkpoint every N steps")

    # --- Constrained MORL (Issue #6, DESIGN-constrained.md) ---
    # All defaults keep use_lagrangian=False so existing pilots stay bit-identical.
    parser.add_argument("--use_lagrangian", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--constraint_handler", type=str, default="lagrangian",
                        choices=["lagrangian", "barrier", "projection"],
                        help="Constraint-handling scheme. PR-B ships only 'lagrangian'.")
    parser.add_argument("--A_max", type=float, default=3.0,
                        help="AoSI tail bound: P[max_k A_k > A_max] <= epsilon_aosi")
    parser.add_argument("--epsilon_aosi", type=float, default=0.05,
                        help="Allowed tail-violation rate for c_1")
    parser.add_argument("--E_total_kJ", type=float, default=30.0,
                        help="Per-episode fleet energy budget in kJ. Caller scales for M>=2 if desired.")
    parser.add_argument("--rho_min", type=float, default=0.7,
                        help="Minimum service rate floor (rolling window) for c_3")
    parser.add_argument("--service_window", type=int, default=20,
                        help="Sliding-window length (slots) for c_3 estimate")
    parser.add_argument("--lambda_lr", type=float, default=1e-3,
                        help="Dual ascent learning rate alpha_lambda")
    parser.add_argument("--lambda_max", type=float, default=100.0,
                        help="Cap on each lambda_i to prevent runaway")
    parser.add_argument("--lambda_init", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        help="Initial dual variables (3-vector: aosi_tail, energy_budget, service_rate)")
    parser.add_argument("--dual_update_every", type=int, default=1000,
                        help="Env steps between outer-loop dual updates")
    parser.add_argument("--ema_decay", type=float, default=0.95,
                        help="EMA decay beta for cost smoothing in dual update")

    # --- wandb ---
    parser.add_argument("--wandb_project", type=str, default="COLA-SemCom")
    parser.add_argument("--wandb_offline", action="store_true", default=False)

    args = parser.parse_args()

    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    # Multi-UAV routing: M >= 2 forces the multi-UAV env even if --env_id
    # was left at the default. Single-UAV path stays unchanged.
    if args.num_uavs >= 2:
        env_id_used = "UAV-SemCom-Multi-v0"
        env = gym.make(
            env_id_used,
            num_uavs=args.num_uavs,
            num_devices=args.num_devices,
            area_size=args.area_size,
            uav_height=args.uav_height,
            max_episode_steps=args.max_episode_steps,
            device_mobility=args.device_mobility,
            device_speed=args.device_speed,
        )
    else:
        env_id_used = args.env_id
        env = gym.make(
            env_id_used,
            num_devices=args.num_devices,
            area_size=args.area_size,
            uav_height=args.uav_height,
            max_episode_steps=args.max_episode_steps,
            device_mobility=args.device_mobility,
            device_speed=args.device_speed,
        )

    name = (
        f"{'MAC' if args.num_uavs >= 2 else 'COLA'}-SemMORL"
        f"_M{args.num_uavs}_dev{args.num_devices}"
        f"_COR-a{args.regular_alpha}_bar{args.regular_bar}"
        f"_lat{args.latent_dim}"
        f"_seed{args.seed}"
    )

    configs = {
        "num_steps": args.num_steps,
        "batch_size": args.batch_size,
        "lr": 0.0003,
        "hidden_units": [256, 256],
        "memory_size": 1e6,
        "prefer_num": args.prefer,
        "buf_num": args.buf_num,
        "gamma": args.gamma,
        "tau": 0.005,
        "entropy_tuning": args.entropy_tuning,
        "ent_coef": args.ent_coef,
        "multi_step": 1,
        "per": False,
        "alpha": 0.6,
        "beta": 0.4,
        "beta_annealing": 0.0001,
        "grad_clip": None,
        "updates_per_step": 1,
        "start_steps": 10000,
        "log_interval": 10,
        "target_update_interval": 1,
        "eval_interval": args.eval_interval,
        "cuda": args.cuda,
        "seed": args.seed,
        "cuda_device": args.cuda_device,
        "q_frequency": args.q_freq,
        "model_saved_step": args.model_saved_step,
        "Use_Policy_Preference": args.Use_Policy_Preference,
        "Use_Critic_Preference": args.Use_Critic_Preference,
        "train_with_fixed_preference": args.train_with_fixed_preference,
        "iso_sigma": 0.005,
        "line_sigma": 0.05,
        "EA_policy_num": args.EA_policy_num,
        "warm_steps": args.warm_steps,
        "RL_policy_num": args.RL_policy_num,
        "latent_dim": args.latent_dim,
        "reward_coef": 1.0,
        "dynamic_coef": 1.0,
        "value_coef": 1.0,
        "Policy_use_latent": args.Policy_use_latent,
        "Policy_use_s": args.Policy_use_s,
        "Policy_use_w": args.Policy_use_w,
        "Critic_use_s": args.Critic_use_s,
        "Critic_use_a": args.Critic_use_a,
        "Policy_use_target": args.Policy_use_target,
        "encoder_update_freq": args.encoder_update_freq,
        "use_avg": args.use_avg,
        "Critic_use_both": args.Critic_use_both,
        "use_encoder_hardupdate": args.use_encoder_hardupdate,
        "regular_alpha": args.regular_alpha,
        "Wandb_name": name,
        "Use_pc_grad": args.Use_pc_grad,
        "step_random": args.step_random,
        "old_Q_update_freq": args.old_Q_update_freq,
        "regular_bar": args.regular_bar,
        "consider_other": args.consider_other,
        "fixed_weight": args.fixed_weight,
        "use_lagrangian": args.use_lagrangian,
        "constraint_handler": args.constraint_handler,
        "constraint_thresholds": {
            "A_max": args.A_max,
            "epsilon_aosi": args.epsilon_aosi,
            "E_total_kJ": args.E_total_kJ,
            "rho_min": args.rho_min,
            "service_window": args.service_window,
        },
        "lambda_lr": args.lambda_lr,
        "lambda_max": args.lambda_max,
        "lambda_init": args.lambda_init,
        "dual_update_every": args.dual_update_every,
        "ema_decay": args.ema_decay,
    }

    exp_tag = args.exp_name or f"COLA-SemCom-seed{args.seed}_dev{args.num_devices}"
    log_dir = os.path.join("logs", "uav", exp_tag)

    our_wandb = wandb.init(project=args.wandb_project, name=name, config=configs)
    if args.num_uavs >= 2:
        from multi_agent import MultiAgentSemMORL
        agent = MultiAgentSemMORL(
            env_id=env_id_used, env=env, log_dir=log_dir,
            num_uavs=args.num_uavs, num_devices=args.num_devices, **configs,
        )
    else:
        agent = SacAgent(env_id=env_id_used, env=env, log_dir=log_dir, **configs)
    agent.run(our_wandb)


if __name__ == "__main__":
    run()
