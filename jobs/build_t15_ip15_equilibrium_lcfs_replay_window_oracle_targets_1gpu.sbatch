#!/bin/bash
#SBATCH --job-name=tokamak-build-ip15-equilibrium-lcfs
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/tokamak-build-ip15-equilibrium-lcfs-%j.out
#SBATCH --error=slurm_logs/tokamak-build-ip15-equilibrium-lcfs-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak/tokamak-rl-v2
mkdir -p slurm_logs data/processed

export READINESS_ROOT=${READINESS_ROOT:-/workspace/tokamak-sim/runs/t15_compact_gpu_batch32_readiness_${SLURM_JOB_ID}}
export READINESS_JSON=${READINESS_JSON:-${READINESS_ROOT}/batch32_readiness.json}
export REPLAY_ROOT=${REPLAY_ROOT:-/workspace/tokamak-sim/runs/t15md_trim50_ip15_equilibrium_lcfs_dev8_replay_gpu_full_${SLURM_JOB_ID}}
export TARGET_DIR=${TARGET_DIR:-data/processed/t15_ip15_equilibrium_lcfs_replay_window_0p1s_oracle_targets}
export RESET_LIBRARY=${RESET_LIBRARY:-data/processed/t15_ip15_equilibrium_lcfs_replay_window_0p1s_oracle_initial_states.npz}
export CONFIG_OUT=${CONFIG_OUT:-configs/experiments/t15_ip15_equilibrium_lcfs_replay_window_0p1s_tcvjdot_mpo.yaml}

printf 'readiness_root=%s\n' "$READINESS_ROOT"
printf 'readiness_json=%s\n' "$READINESS_JSON"
printf 'replay_root=%s\n' "$REPLAY_ROOT"
printf 'target_dir=%s\n' "$TARGET_DIR"
printf 'reset_library=%s\n' "$RESET_LIBRARY"
printf 'config_out=%s\n' "$CONFIG_OUT"

srun \
  --ntasks=1 \
  --container-image /scratch/$USER/tokamak/tokamak-rl-v2.sqsh \
  --container-mounts /scratch/$USER/tokamak:/workspace,/dev:/dev,/lib/x86_64-linux-gnu:/host-libs \
  bash -lc '
    set -euo pipefail
    mkdir -p /tmp/nvidia-lib
    ln -sf /host-libs/libcuda.so.1 /tmp/nvidia-lib/libcuda.so.1
    ln -sf /tmp/nvidia-lib/libcuda.so.1 /tmp/nvidia-lib/libcuda.so
    ln -sf /host-libs/libnvidia-ml.so.1 /tmp/nvidia-lib/libnvidia-ml.so.1
    ln -sf /tmp/nvidia-lib/libnvidia-ml.so.1 /tmp/nvidia-lib/libnvidia-ml.so
    export LD_LIBRARY_PATH=/tmp/nvidia-lib:${LD_LIBRARY_PATH:-}
    export PYTHONPATH=/workspace/tokamak-sim:/workspace/tokamak-rl-v2:${PYTHONPATH:-}

    cd /workspace/tokamak-sim
    echo "===== server stage 1/3: compact GPU batch=32 readiness ====="
    python3 scripts/run_t15md_limited_replay_dataset.py \
      --shots 3857 \
      --config configs/T15MD.toml \
      --data-root data/t15_data_new_trim50_ip_calibrated \
      --raw-data-root data/t15_data_new \
      --initial-prefix T15MD_batch32_readiness \
      --out "$READINESS_ROOT" \
      --angles 32 \
      --boundary-mode equilibrium_lcfs \
      --compute-backend gpu \
      --gpu-device cuda:0 \
      --no-video \
      --dry-run

    python3 scripts/validate_t15_compact_gpu_batch32.py \
      --config configs/T15MD.toml \
      --initial-state "$READINESS_ROOT/initial_states/T15MD_batch32_readiness_3857.toml" \
      --coils data/t15_data_new_trim50_ip_calibrated/coils/t15md_3857_coils.csv \
      --gpu-device cuda:0 \
      --batch-size 32 \
      --warmup-steps 4 \
      --benchmark-steps 64 \
      --json-out "$READINESS_JSON"

    python3 - <<"PYREADINESS"
import json
import os
from pathlib import Path

summary = json.loads(Path(os.environ["READINESS_JSON"]).read_text(encoding="utf-8"))
required_true = (
    "accepted",
    "reset_indices_unaffected_lanes_exact",
    "identical_lane_state_parity",
)
for key in required_true:
    if summary.get(key) is not True:
        raise RuntimeError(f"batch=32 readiness failed: {key}={summary.get(key)!r}")
if int(summary.get("batch_size", -1)) != 32:
    raise RuntimeError(f"batch=32 readiness returned batch_size={summary.get('batch_size')!r}")
if int(summary.get("simulator_hot_path_host_transfer_tokens", -1)) != 0:
    raise RuntimeError("batch=32 readiness found host-transfer tokens in the simulator hot path")
for key in ("simulator_steps_per_second", "environment_steps_per_second", "peak_allocated_mib", "peak_reserved_mib"):
    value = float(summary.get(key, float("nan")))
    if not (value > 0.0):
        raise RuntimeError(f"batch=32 readiness produced invalid {key}={value!r}")
print("batch32_readiness_accepted", json.dumps(summary, sort_keys=True), flush=True)
PYREADINESS

    echo "===== server stage 2/3: fresh eight-shot development canonical GPU replay ====="
    python3 scripts/run_t15md_limited_replay_dataset.py \
      --shots 3854 3855 3856 3857 3858 3859 3862 3863 \
      --config configs/T15MD.toml \
      --data-root data/t15_data_new_trim50_ip_calibrated \
      --raw-data-root data/t15_data_new \
      --initial-prefix T15MD_ip15_equilibrium_lcfs \
      --out "$REPLAY_ROOT" \
      --angles 32 \
      --boundary-mode equilibrium_lcfs \
      --compute-backend gpu \
      --gpu-device cuda:0 \
      --no-video

    python3 - <<"PYREPLAYCHECK"
import os
from pathlib import Path

import numpy as np

root = Path(os.environ["REPLAY_ROOT"])
expected_shots = {"3854", "3855", "3856", "3857", "3858", "3859", "3862", "3863"}
required = {
    "version",
    "radii_true",
    "boundary_poly_true",
    "boundary_topology_code",
    "boundary_axis",
    "boundary_x_points",
    "boundary_limiter_contacts",
    "boundary_fixed_angle_valid",
    "boundary_fixed_angle_counts",
    "boundary_quality",
}
artifacts_by_shot: dict[str, list[Path]] = {shot: [] for shot in expected_shots}

for path in sorted(root.glob("t15md_limited_replay_*/**/run*.npz")):
    shot = next(
        (
            part.rsplit("_", 1)[-1]
            for part in path.parts
            if part.startswith("t15md_limited_replay_")
        ),
        None,
    )
    if shot in artifacts_by_shot:
        artifacts_by_shot[shot].append(path)

artifact_counts = {shot: len(paths) for shot, paths in artifacts_by_shot.items()}
if any(count != 1 for count in artifact_counts.values()):
    raise RuntimeError(
        "expected exactly one fresh replay artifact per shot, "
        f"got {artifact_counts} under {root}"
    )

for shot in sorted(expected_shots, key=int):
    path = artifacts_by_shot[shot][0]
    with np.load(path, allow_pickle=False) as run:
        missing = sorted(required - set(run.files))
        if missing:
            raise RuntimeError(f"{path}: missing schema-v4 keys {missing}")

        version = np.asarray(run["version"]).reshape(-1)
        if version.size != 1 or int(version[0]) != 4:
            raise RuntimeError(
                f"{path}: expected version=4, got {version.tolist()}"
            )

        radii = np.asarray(run["radii_true"], dtype=float)
        polygon = np.asarray(run["boundary_poly_true"], dtype=float)
        topology = np.asarray(run["boundary_topology_code"], dtype=np.int64).reshape(-1)
        axis = np.asarray(run["boundary_axis"], dtype=float)
        x_points = np.asarray(run["boundary_x_points"], dtype=float)
        contacts = np.asarray(run["boundary_limiter_contacts"], dtype=float)
        fixed_valid = np.asarray(
            run["boundary_fixed_angle_valid"],
            dtype=bool,
        ).reshape(-1)
        counts = np.asarray(
            run["boundary_fixed_angle_counts"],
            dtype=np.int64,
        )
        quality = np.asarray(run["boundary_quality"], dtype=float)

        if radii.ndim != 2 or radii.shape[1] != 32:
            raise RuntimeError(
                f"{path}: expected radii_true shape (steps, 32), got {radii.shape}"
            )
        steps = int(radii.shape[0])
        expected_shapes = {
            "boundary_fixed_angle_valid": (steps,),
            "boundary_fixed_angle_counts": (steps, 32),
            "boundary_topology_code": (steps,),
            "boundary_axis": (steps, 2),
            "boundary_quality": (steps, 6),
        }
        actual_shapes = {
            "boundary_fixed_angle_valid": fixed_valid.shape,
            "boundary_fixed_angle_counts": counts.shape,
            "boundary_topology_code": topology.shape,
            "boundary_axis": axis.shape,
            "boundary_quality": quality.shape,
        }
        mismatched = {
            key: {"expected": expected_shapes[key], "actual": actual_shapes[key]}
            for key in expected_shapes
            if actual_shapes[key] != expected_shapes[key]
        }
        if mismatched:
            raise RuntimeError(f"{path}: LCFS channel shape mismatch {mismatched}")
        if polygon.ndim != 3 or polygon.shape[0] != steps or polygon.shape[2] != 2:
            raise RuntimeError(f"{path}: invalid boundary_poly_true shape {polygon.shape}")
        if x_points.ndim != 3 or x_points.shape[0] != steps or x_points.shape[2] != 2:
            raise RuntimeError(f"{path}: invalid boundary_x_points shape {x_points.shape}")
        if contacts.ndim != 3 or contacts.shape[0] != steps or contacts.shape[2] != 2:
            raise RuntimeError(
                f"{path}: invalid boundary_limiter_contacts shape {contacts.shape}"
            )

        radii_ok = np.isfinite(radii).all(axis=1) & (radii > 0.0).all(axis=1)
        polygon_finite = np.isfinite(polygon).all(axis=2)
        polygon_points = polygon_finite.sum(axis=1)
        polygon_last = polygon[np.arange(steps), polygon_points - 1, :]
        polygon_closed = (
            np.linalg.norm(polygon[:, 0, :] - polygon_last, axis=1) <= 1.0e-8
        )
        topology_ok = np.isin(topology, np.asarray([1, 2, 3, 4]))
        counts_ok = (counts == 1).all(axis=1)
        axis_ok = np.isfinite(axis).all(axis=1)
        quality_ok = np.isfinite(quality).all(axis=1)
        x_count = np.isfinite(x_points).all(axis=2).sum(axis=1)
        contact_count = np.isfinite(contacts).all(axis=2).sum(axis=1)
        topology_detail_ok = (
            ((topology == 1) & (x_count == 0) & (contact_count >= 1))
            | ((topology == 2) & (x_count == 1))
            | ((topology == 3) & (x_count == 2))
            | ((topology == 4) & (x_count >= 3))
        )

        invalid = np.flatnonzero(
            ~fixed_valid
            | ~radii_ok
            | (polygon_points < 4)
            | ~polygon_closed
            | ~topology_ok
            | ~counts_ok
            | ~axis_ok
            | ~quality_ok
            | ~topology_detail_ok
        )
        if invalid.size:
            preview = invalid[:20].tolist()
            raise RuntimeError(
                f"{path}: inconsistent full-GPU LCFS artifacts at steps {preview}; "
                f"invalid_count={int(invalid.size)}"
            )

        reference_path = root / f"lqr_boundary_reference_{shot}.npz"
        if not reference_path.exists():
            raise RuntimeError(f"missing replay reference {reference_path}")
        with np.load(reference_path, allow_pickle=False) as ref:
            ref_required = {
                "schema",
                "source_index",
                "t",
                "Ip",
                "radii_true",
                "boundary_found",
                "boundary_fixed_angle_valid",
                "pfc_currents",
                "sol_currents",
                "hidden_state_a",
                "passive_currents_a",
            }
            ref_missing = sorted(ref_required - set(ref.files))
            if ref_missing:
                raise RuntimeError(
                    f"{reference_path}: missing replay reference fields {ref_missing}"
                )
            ref_schema = np.asarray(ref["schema"]).astype(str).reshape(-1)
            ref_source_index = np.asarray(ref["source_index"], dtype=np.int64).reshape(-1)
            ref_t = np.asarray(ref["t"], dtype=float).reshape(-1)
            ref_ip = np.asarray(ref["Ip"], dtype=float).reshape(-1)
            ref_radii = np.asarray(ref["radii_true"], dtype=float)
            ref_found = np.asarray(ref["boundary_found"], dtype=bool).reshape(-1)
            ref_projection = np.asarray(ref["boundary_fixed_angle_valid"], dtype=bool).reshape(-1)
            ref_pfc = np.asarray(ref["pfc_currents"], dtype=float)
            ref_sol = np.asarray(ref["sol_currents"], dtype=float)
            ref_hidden = np.asarray(ref["hidden_state_a"], dtype=float).reshape(-1)
            ref_passive = np.asarray(ref["passive_currents_a"], dtype=float)
        endpoints = steps + 1
        if ref_schema.shape != (1,) or ref_schema[0] != "t15md_replay_reference_v2":
            raise RuntimeError(f"{reference_path}: unexpected schema {ref_schema.tolist()}")
        if not np.array_equal(ref_source_index, np.arange(endpoints, dtype=np.int64)):
            raise RuntimeError(f"{reference_path}: source_index is not endpoint-aligned 0..{endpoints - 1}")
        if ref_t.shape != (endpoints,) or ref_ip.shape != (endpoints,):
            raise RuntimeError(
                f"{reference_path}: replay reference length mismatch "
                f"t={ref_t.shape} Ip={ref_ip.shape} expected={(endpoints,)}"
            )
        if ref_radii.shape != (endpoints, 32):
            raise RuntimeError(f"{reference_path}: radii shape {ref_radii.shape} != {(endpoints, 32)}")
        if ref_found.shape != (endpoints,) or ref_projection.shape != (endpoints,):
            raise RuntimeError(f"{reference_path}: boundary validity arrays are not endpoint-aligned")
        if ref_hidden.shape != (endpoints,):
            raise RuntimeError(f"{reference_path}: hidden_state_a shape {ref_hidden.shape} != {(endpoints,)}")
        if ref_passive.shape != (endpoints, 303):
            raise RuntimeError(f"{reference_path}: passive_currents_a shape {ref_passive.shape} != {(endpoints, 303)}")
        if ref_pfc.shape != (endpoints, 6) or ref_sol.shape != (endpoints, 3):
            raise RuntimeError(f"{reference_path}: coil endpoint arrays have wrong shapes")
        if not np.all(ref_found):
            raise RuntimeError(f"{reference_path}: boundary_found is not 100%")
        if not np.all(ref_projection):
            raise RuntimeError(f"{reference_path}: projection_valid is not 100%")
        if not np.all(np.isfinite(ref_hidden)) or not np.all(np.isfinite(ref_passive)):
            raise RuntimeError(f"{reference_path}: non-finite compact/passive reset state")
        if not np.array_equal(ref_radii[1:], radii, equal_nan=True):
            raise RuntimeError(
                f"{reference_path}: post-reset radii_true differs from source run artifact {path}"
            )
        if not np.allclose(ref_ip[1:], np.asarray(run["Ip"], dtype=float), rtol=0.0, atol=1.0e-8):
            raise RuntimeError(f"{reference_path}: post-reset Ip differs from source run artifact {path}")

        print(
            "validated_full_gpu_replay",
            shot,
            "steps",
            steps,
            "path",
            path,
            "reference",
            reference_path,
        )
PYREPLAYCHECK

    cd /workspace/tokamak-rl-v2
    echo "===== server stage 3/3: replay-window oracle/reset dataset ====="
    python3 scripts/build_t15_new_replay_window_oracle_targets.py \
      --base-config configs/experiments/t15_ip15_equilibrium_lcfs_replay_window_0p1s_tcvjdot_mpo_balanced.yaml \
      --data-root ../tokamak-sim/data/t15_data_new_trim50_ip_calibrated \
      --machine-config ../tokamak-sim/configs/T15MD.toml \
      --replay-root "$REPLAY_ROOT" \
      --target-dir "$TARGET_DIR" \
      --initial-library-out "$RESET_LIBRARY" \
      --config-out "$CONFIG_OUT" \
      --window-steps 100 \
      --angles 32 \
      --compute-backend gpu \
      --gpu-device cuda:0 \
      --train-shots 3854 3855 3856 3857 3858 3859 3862 3863

    python3 - <<"PYCHECK"
import json
import os
from pathlib import Path

import numpy as np

summary_path = Path(os.environ["TARGET_DIR"]) / "oracle_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
expected_shots = {"3854", "3855", "3856", "3857", "3858", "3859", "3862", "3863"}
actual_shots = set(summary["accepted_by_shot"])
if actual_shots != expected_shots:
    raise SystemExit(
        "oracle dataset shots mismatch: "
        f"expected={sorted(expected_shots)} actual={sorted(actual_shots)}"
    )

data_root = Path("../tokamak-sim/data/t15_data_new_trim50_ip_calibrated")
window_steps = 100
expected_by_shot = {}
for shot in sorted(expected_shots, key=int):
    ip_path = data_root / "ip" / f"t15md_{shot}_ip.csv"
    rows = sum(1 for line in ip_path.read_text(encoding="utf-8").splitlines() if line.strip())
    expected_by_shot[shot] = rows - window_steps
    if expected_by_shot[shot] <= 0:
        raise SystemExit(f"shot {shot} has no complete {window_steps}-step windows")
expected_windows = sum(expected_by_shot.values())
actual_by_shot = {str(key): int(value) for key, value in summary["accepted_by_shot"].items()}
if actual_by_shot != expected_by_shot:
    raise SystemExit(
        f"oracle accepted_by_shot mismatch: expected={expected_by_shot} actual={actual_by_shot}"
    )
if int(summary.get("accepted_windows", -1)) != expected_windows:
    raise SystemExit(
        f"oracle accepted window count mismatch: expected={expected_windows} "
        f"actual={summary.get('accepted_windows')}"
    )
if int(summary.get("rejected_windows", -1)) != 0:
    raise SystemExit(
        f"fresh canonical oracle rejected {summary.get('rejected_windows')} windows; expected zero"
    )
if summary.get("train_shots") != ["3854", "3855", "3856", "3857", "3858", "3859", "3862", "3863"]:
    raise SystemExit(f"oracle train shots mismatch: {summary.get('train_shots')}")
if summary.get("holdout_shots") != []:
    raise SystemExit(f"oracle holdout shots mismatch: {summary.get('holdout_shots')}")
if int(summary.get("window_steps", -1)) != 100 or int(summary.get("angles", -1)) != 32:
    raise SystemExit(
        f"oracle representation mismatch: window_steps={summary.get('window_steps')} "
        f"angles={summary.get('angles')}"
    )
if summary.get("source_replay_root") != os.environ["REPLAY_ROOT"]:
    raise SystemExit(
        f"oracle source replay mismatch: {summary.get('source_replay_root')} "
        f"!= {os.environ['REPLAY_ROOT']}"
    )
contract = summary.get("boundary_contract", {})
if contract.get("boundary_found_required_all_steps") is not True:
    raise SystemExit("oracle summary does not enforce complete boundary_found")
if contract.get("projection_valid_required_all_steps") is not True:
    raise SystemExit("oracle summary does not enforce complete projection_valid")
if contract.get("fixed_angle_radii_required_all_steps") is not True:
    raise SystemExit("oracle summary does not enforce complete fixed-angle radii")
reset_contract = summary.get("reset_contract", {})
if reset_contract.get("source_index_endpoint_aligned") is not True:
    raise SystemExit("oracle reset contract is not source-index endpoint aligned")
if reset_contract.get("compact_hidden_state_required") is not True:
    raise SystemExit("oracle reset contract does not require compact hidden state")
if reset_contract.get("physical_passive_currents_required") is not True:
    raise SystemExit("oracle reset contract does not require physical passive currents")
if reset_contract.get("future_measured_ip_after_reset_endpoint_used") is not False:
    raise SystemExit("oracle reset contract permits future measured Ip leakage")
if summary.get("oracle_ip_error_filtering") is not False:
    raise SystemExit("fresh replay oracle must record Ip error as a metric, not filter windows by it")

reset_path = Path(os.environ["RESET_LIBRARY"])
with np.load(reset_path, allow_pickle=False) as reset:
    reset_required = {
        "schema", "shot_id", "source_index", "time_s", "ip0", "pfc0", "sol0",
        "split", "difficulty_bin", "hidden_state_a", "passive_currents_a"
    }
    missing = sorted(reset_required - set(reset.files))
    if missing:
        raise SystemExit(f"reset library missing canonical fields: {missing}")
    reset_schema = np.asarray(reset["schema"]).astype(str).reshape(-1)
    reset_ip = np.asarray(reset["ip0"], dtype=float).reshape(-1)
    reset_hidden = np.asarray(reset["hidden_state_a"], dtype=float).reshape(-1)
    reset_passive = np.asarray(reset["passive_currents_a"], dtype=float)
    reset_pfc = np.asarray(reset["pfc0"], dtype=float)
    reset_sol = np.asarray(reset["sol0"], dtype=float)
if reset_schema.shape != (1,) or reset_schema[0] != "t15_replay_window_oracle_initial_states_v2":
    raise SystemExit(f"unexpected reset schema: {reset_schema.tolist()}")
rows = int(reset_ip.size)
if reset_hidden.shape != (rows,) or reset_passive.shape != (rows, 303):
    raise SystemExit(
        f"reset compact/passive shapes invalid hidden={reset_hidden.shape} passive={reset_passive.shape} rows={rows}"
    )
if reset_pfc.shape != (rows, 6) or reset_sol.shape != (rows, 3):
    raise SystemExit(f"reset coil shapes invalid pfc={reset_pfc.shape} sol={reset_sol.shape}")
if not np.all(np.isfinite(reset_ip)) or not np.all(np.isfinite(reset_hidden)) or not np.all(np.isfinite(reset_passive)):
    raise SystemExit("reset library contains non-finite compact/passive state")
if int(summary.get("accepted_windows", -1)) != rows:
    raise SystemExit(f"accepted/reset row count mismatch summary={summary.get('accepted_windows')} reset={rows}")

if int(summary["split_counts"].get("holdout", 0)) <= 0:
    raise SystemExit("oracle dataset has no holdout windows")
if int(summary["split_counts"].get("train", 0)) <= 0:
    raise SystemExit("oracle dataset has no train windows")

readiness = json.loads(Path(os.environ["READINESS_JSON"]).read_text(encoding="utf-8"))
acceptance = {
    "schema": "t15_compact_gpu_server_dataset_acceptance_v1",
    "accepted": True,
    "batch32": readiness,
    "replay_root": os.environ["REPLAY_ROOT"],
    "oracle_summary": str(summary_path),
    "expected_windows": expected_windows,
    "accepted_windows": int(summary["accepted_windows"]),
    "rejected_windows": int(summary["rejected_windows"]),
    "accepted_by_shot": actual_by_shot,
    "boundary_found_fraction_required": 1.0,
    "projection_valid_fraction_required": 1.0,
}
acceptance_path = Path(os.environ["TARGET_DIR"]) / "server_stage_acceptance.json"
acceptance_path.write_text(json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"server_stage_acceptance={acceptance_path}")
print(json.dumps(acceptance, indent=2, sort_keys=True))
PYCHECK
  '
