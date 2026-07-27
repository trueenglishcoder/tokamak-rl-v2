"""Замена _simulate_gpu — читает готовые реплеи вместо пересимуляции."""
from __future__ import annotations
from pathlib import Path
import numpy as np


def load_oracle_from_replay(
    candidates: list,
    *,
    replay_root: Path,
    angles: int,
    mean_ip_limit: float,
    max_ip_limit: float,
    n_pfc: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Загружает окна из готовых NPZ реплеев вместо GPU-симуляции.
    
    Args:
        candidates: список WindowCandidate
        replay_root: путь к папке с реплеями (runs/top5_spline_replay)
        angles: число углов
        mean_ip_limit: лимит средней ошибки Ip для acceptance
        max_ip_limit: лимит максимальной ошибки Ip
        n_pfc: число PFC катушек
    
    Returns:
        accepted, rejected — списки словарей как в _simulate_gpu
    """
    import re
    
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    
    # Кеш загруженных NPZ по shot_id
    npz_cache: dict[str, dict] = {}
    
    # Паттерн для поиска run*.npz в поддиректории
    # Структура: t15md_limited_replay_XXXX/YYYY*/run*.npz
    
    for cand in candidates:
        shot = str(cand.shot_id)
        
        # Загружаем NPZ если ещё не в кеше
        if shot not in npz_cache:
            replay_dir = replay_root / f"t15md_limited_replay_{shot}"
            if not replay_dir.exists():
                # пробуем lqr_boundary_reference
                lqr_path = replay_root / f"lqr_boundary_reference_{shot}.npz"
                if lqr_path.exists():
                    npz_cache[shot] = dict(np.load(lqr_path, allow_pickle=False))
                    npz_cache[shot]["_t"] = npz_cache[shot]["t"]
                    npz_cache[shot]["_Ip"] = npz_cache[shot]["Ip"]
                    npz_cache[shot]["_radii"] = npz_cache[shot]["radii_true"]
                    npz_cache[shot]["_found"] = npz_cache[shot].get("boundary_found", np.ones(len(npz_cache[shot]["t"]), dtype=bool))
                    npz_cache[shot]["_pfc"] = npz_cache[shot]["pfc_currents"]
                    npz_cache[shot]["_sol"] = npz_cache[shot]["sol_currents"]
                    continue
            
            # Ищем run*.npz внутри поддиректорий
            run_npz = list(replay_dir.glob("*/run*.npz"))
            if not run_npz:
                rejected.append(_reject(cand, "no_replay_npz"))
                continue
            # Берём последний по дате
            run_npz.sort()
            data = dict(np.load(run_npz[-1], allow_pickle=False))
            
            # Извлекаем нужные поля
            t = data["t"].reshape(-1)
            ip = data.get("Ip", data.get("Ip_ref", None))
            if ip is None:
                rejected.append(_reject(cand, "no_Ip_in_npz"))
                continue
            ip = ip.reshape(-1)
            
            radii = data.get("radii_true", data.get("radii_ref", None))
            if radii is None:
                rejected.append(_reject(cand, "no_radii_in_npz"))
                continue
            
            found = data.get("boundary_found", np.ones(len(t), dtype=bool)).reshape(-1)
            pfc = data.get("pfc_currents", data.get("pfc_currents_cmd", None))
            sol = data.get("sol_currents", data.get("sol_currents_cmd", None))
            
            npz_cache[shot] = {
                "_t": t,
                "_Ip": ip,
                "_radii": radii,
                "_found": found,
                "_pfc": pfc,
                "_sol": sol,
            }
        
        cached = npz_cache[shot]
        t = cached["_t"]
        ip_full = cached["_Ip"]
        radii_full = cached["_radii"]
        found_full = cached["_found"]
        
        # Извлекаем окно [source_index : source_index + window_steps + 1]
        start = int(cand.source_index)
        steps = cand.ip_target.shape[0]  # window_steps + 1
        
        if start + steps > len(t):
            rejected.append(_reject(cand, f"window_out_of_bounds_{start}+{steps}>{len(t)}"))
            continue
        
        window_ip = ip_full[start : start + steps]
        window_radii = radii_full[start : start + steps]
        window_found = found_full[start : start + steps]
        
        # Проверки
        if not np.all(np.isfinite(window_ip)):
            rejected.append(_reject(cand, "non_finite_ip"))
            continue
        if not np.all(window_found):
            rejected.append(_reject(cand, "boundary_not_found"))
            continue
        if window_radii.shape[1] != angles:
            rejected.append(_reject(cand, f"wrong_angles_{window_radii.shape[1]}_expected_{angles}"))
            continue
        
        # Oracle IP error
        ip_error = np.abs(window_ip - cand.ip_target)
        mean_err = float(np.mean(ip_error))
        max_err = float(np.max(ip_error))
        
        if mean_err > mean_ip_limit:
            rejected.append(_reject(cand, f"oracle_mean_ip_{mean_err:.0f}A"))
            continue
        if max_err > max_ip_limit:
            rejected.append(_reject(cand, f"oracle_max_ip_{max_err:.0f}A"))
            continue
        
        # Вычисляем oracle action (производные токов катушек)
        if cached["_pfc"] is not None and cached["_sol"] is not None:
            pfc_window = cached["_pfc"][start : start + steps]
            sol_window = cached["_sol"][start : start + steps]
            currents = np.concatenate([pfc_window, sol_window], axis=1)
            dt = float(t[1] - t[0]) if len(t) > 1 else 0.001
            real_jdot = np.diff(currents, axis=0) / dt
        else:
            real_jdot = cand.real_jdot  # fallback к табличным
        
        n_pfc_local = n_pfc
        pfc0 = cand.currents[0, :n_pfc_local].astype(np.float32)
        sol0 = cand.currents[0, n_pfc_local:].astype(np.float32)
        ip0 = float(cand.ip_target[0])
        boundary_radii = np.asarray(window_radii, dtype=np.float32)  # steps 1..T (without t=0)
        real_jdot_action = cand.normalized_action.astype(np.float32)
        
        accepted.append({
            "shot_id": str(shot),
            "split": cand.split,
            "source_index": int(start),
            "time_s": float(t[start]),
            "difficulty_bin": cand.difficulty_bin,
            "ip0": ip0,
            "pfc0": pfc0,
            "sol0": sol0,
            "ip_target": cand.ip_target.astype(np.float32),
            "boundary_radii": boundary_radii,
            "real_jdot_action": real_jdot_action,
            "oracle_ip_mean_error_a": np.float32(mean_err),
            "oracle_ip_max_error_a": np.float32(max_err),
        })
    
    return accepted, rejected


def _reject(cand, reason: str) -> dict[str, object]:
    return {
        "shot": str(cand.shot_id),
        "source_index": int(cand.source_index),
        "time_s": float(cand.time_s),
        "reason": reason,
    }

