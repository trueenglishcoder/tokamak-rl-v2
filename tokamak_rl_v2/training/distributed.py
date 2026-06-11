from __future__ import annotations

import queue
import time
from dataclasses import replace
from multiprocessing import Event, Queue, get_context
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tokamak_rl_v2.config.schema import ExperimentConfig
from tokamak_rl_v2.env import TokamakMagneticControlEnv
from tokamak_rl_v2.networks import FeedForwardGaussianActor


def start_actor_workers(
    *,
    config: ExperimentConfig,
    actor_state_dict: dict[str, torch.Tensor],
    worker_count: int,
    envs_per_worker: int,
    rollout_chunk_length: int,
    actor_devices: tuple[str, ...],
    seed: int,
) -> tuple[list[Any], list[Queue], Queue, Event]:
    ctx = get_context("spawn")
    data_q: Queue = ctx.Queue(maxsize=max(2, int(worker_count) * 2))
    stop = ctx.Event()
    param_queues: list[Queue] = []
    procs = []
    for index in range(int(worker_count)):
        device = str(actor_devices[index])
        pq: Queue = ctx.Queue(maxsize=2)
        pq.put({k: v.detach().cpu().numpy() for k, v in actor_state_dict.items()})
        proc = ctx.Process(
            name=f"actor-worker-{index}-{device.replace(':', '_')}",
            target=_actor_loop,
            kwargs={
                "config": config,
                "worker_index": index,
                "envs_per_worker": int(envs_per_worker),
                "rollout_chunk_length": int(rollout_chunk_length),
                "device": str(device),
                "seed": int(seed) + index * 9973,
                "params_q": pq,
                "data_q": data_q,
                "stop": stop,
            },
            daemon=True,
        )
        proc.start()
        param_queues.append(pq)
        procs.append(proc)
    return procs, param_queues, data_q, stop


def broadcast_actor(param_queues: list[Queue], state_dict: dict[str, torch.Tensor]) -> None:
    payload = {k: v.detach().cpu().numpy() for k, v in state_dict.items()}
    for q in param_queues:
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
        q.put(payload)


def stop_actor_workers(processes: list[Any], stop: Event) -> None:
    stop.set()
    for proc in processes:
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.terminate()


def _actor_loop(
    *,
    config: ExperimentConfig,
    worker_index: int,
    envs_per_worker: int,
    rollout_chunk_length: int,
    device: str,
    seed: int,
    params_q: Queue,
    data_q: Queue,
    stop: Event,
) -> None:
    torch.set_num_threads(1)
    dev = _resolve_worker_device(device)
    worker_config = config
    if config.sim.compute_backend == "gpu":
        if dev.type != "cuda":
            raise RuntimeError(f"actor worker {worker_index} requested GPU simulator but actor device is not CUDA: {device}")
        worker_config = replace(config, sim=replace(config.sim, gpu_device=str(dev)))
    env = TokamakMagneticControlEnv(worker_config, batch_size=int(envs_per_worker), device=dev, seed=int(seed))
    actor = FeedForwardGaussianActor(env.obs_dim, env.action_dim, worker_config.network.hidden_dim).to(dev)
    actor.load_state_dict({k: torch.as_tensor(v, device=dev) for k, v in params_q.get().items()})
    obs = env.reset()
    while not stop.is_set():
        _drain_params(actor, params_q, dev)
        chunk = {"obs": [], "action": [], "reward": [], "discount": [], "next_obs": [], "done": []}
        for _ in range(int(rollout_chunk_length)):
            with torch.no_grad():
                action, _logp, _mean = actor.sample(obs)
            out = env.step(action)
            done = out.terminated | out.truncated
            chunk["obs"].append(obs.detach().cpu())
            chunk["action"].append(action.detach().cpu())
            chunk["reward"].append(out.reward.detach().cpu())
            chunk["discount"].append(torch.full_like(out.reward.detach().cpu(), float(worker_config.learner.discount)))
            chunk["next_obs"].append(out.obs.detach().cpu())
            chunk["done"].append(done.detach().cpu())
            obs = env.reset_indices(done) if bool(torch.any(done).item()) else out.obs
        payload = {k: torch.stack(v, dim=0).numpy() for k, v in chunk.items()}
        payload["worker_index"] = int(worker_index)
        payload["worker_device"] = str(dev)
        data_q.put(payload)


def _resolve_worker_device(value: str) -> torch.device:
    dev = torch.device(value)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA actor device requested but torch.cuda.is_available() is false: {value}")
        if dev.index is not None and dev.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA actor device index is not visible: {value}; visible device count is {torch.cuda.device_count()}")
        torch.cuda.set_device(dev)
    return dev


def _drain_params(actor: FeedForwardGaussianActor, params_q: Queue, device: torch.device) -> None:
    latest = None
    try:
        while True:
            latest = params_q.get_nowait()
    except queue.Empty:
        pass
    if latest is not None:
        actor.load_state_dict({k: torch.as_tensor(v, device=device) for k, v in latest.items()})
