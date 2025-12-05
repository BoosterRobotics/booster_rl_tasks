from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Union

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def _get_adaptive_sigma(env, key: str | float, error: Union[float, torch.Tensor]):
    if isinstance(key, float):
        return key
    sigma_update_rate = 0.9
    if not hasattr(env, 'reward_sigmas_ema'):
        env.reward_sigmas_ema = {}
        env.reward_sigmas = {}

    env.reward_sigmas_ema[key] = (
        sigma_update_rate * env.reward_sigmas_ema.get(key, torch.tensor([100.], device=env.device)) + (1 - sigma_update_rate) * error
    )
    env.reward_sigmas[key] = torch.minimum(env.reward_sigmas_ema[key], env.reward_sigmas.get(key, torch.tensor([100.], device=env.device))).clip(min=1e-8)
    return torch.sqrt(env.reward_sigmas[key])


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float | str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    std = _get_adaptive_sigma(env, std, error.mean())
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
        env: ManagerBasedRLEnv, command_name: str, std: float | str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    std = _get_adaptive_sigma(env, std, error.mean())
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float | str, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    ).mean(dim=-1)
    std = _get_adaptive_sigma(env, std, error.mean())
    return torch.exp(-error / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float | str, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    ).mean(dim=-1)
    std = _get_adaptive_sigma(env, std, error.mean())
    return torch.exp(-error / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float | str, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    ).mean(dim=-1)
    std = _get_adaptive_sigma(env, std, error.mean())
    return torch.exp(-error / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float | str, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    ).mean(dim=-1)
    std = _get_adaptive_sigma(env, std, error.mean())
    return torch.exp(-error / std**2)


def feet_stance_time(
        env: ManagerBasedRLEnv, asset_name: str, feet_names: list[str], vel_threshold: float, desired_time: float
) -> torch.Tensor:
    if not hasattr(env, '_buf_feet_stance_time'):
        env._buf_feet_stance_time = torch.zeros(env.num_envs, 2, device=env.device)

    robot = env.scene.articulations[asset_name]
    feet_indexes = [robot.body_names.index(name) for name in feet_names]

    stance = robot.data.body_link_lin_vel_w[:, feet_indexes].norm(dim=-1) < vel_threshold

    first_slide = (env._buf_feet_stance_time > 0.) * (~stance)
    rew_stanceTime = torch.sum((env._buf_feet_stance_time - desired_time).clip(max=0.) * first_slide, dim=1)

    env._buf_feet_stance_time += env.step_dt
    env._buf_feet_stance_time *= stance
    return rew_stanceTime
