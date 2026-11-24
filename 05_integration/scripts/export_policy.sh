#!/usr/bin/env bash
# export_policy.sh — turn a training run into a deployable ONNX policy.
#
#   ./export_policy.sh --source train      --run 04_training/roboparty_train/logs/rsl_rl/rpo_amp/<ts> \
#                      --task RPO-AMP-Play --name amp --install
#   ./export_policy.sh --source ufo        --run 02_motion_infra/UFO/runs/rpo_fb --name ufo --install
#   ./export_policy.sh --source partymimic --run 02_motion_infra/PartyMimic/logs/rp1_s0/ckpt_012000.pt \
#                      --name rp1_mimic
#   ./export_policy.sh --source mimiclite  --run 02_motion_infra/MimicLite/active-adaptation/logs/... \
#                      --task tracking-base --robot-profile atom --name mimic
#
#   --source train|ufo|partymimic|mimiclite
#   --run PATH          run directory (train/ufo/mimiclite) or checkpoint file (partymimic)
#   --checkpoint PATH   explicit checkpoint; default: newest in --run
#   --task NAME         train/mimiclite task id (e.g. RPO-AMP-Play, tracking-base)
#   --robot-profile P   robot profile for mimiclite (default atom)
#   --name NAME         policy name; installed as models/policy_<NAME>.onnx
#   --robot rpo|rp1     deploy robot directory (default rpo)
#   --out PATH          output ONNX path (default: <run>/exported/policy_<name>.onnx)
#   --install           copy the result into roboparty_deploy/<robot>/models (+ sidecars)
#   --verify            print ONNX inputs/outputs and run the shape checks
#   --dry-run           print commands only

set -uo pipefail
# shellcheck source=lib/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

SOURCE=""
RUN=""
CHECKPOINT=""
TASK=""
ROBOT_PROFILE="atom"
NAME=""
ROBOT="rpo"
OUT=""
INSTALL=0
VERIFY=0
MOTION_LIST="${MOTION_LIST:-0}"

usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"; exit "${1:-0}"; }

while (($#)); do
  case "$1" in
    --source) SOURCE="${2:?}"; shift 2 ;;
    --run) RUN="${2:?}"; shift 2 ;;
    --checkpoint) CHECKPOINT="${2:?}"; shift 2 ;;
    --task) TASK="${2:?}"; shift 2 ;;
    --robot-profile) ROBOT_PROFILE="${2:?}"; shift 2 ;;
    --name) NAME="${2:?}"; shift 2 ;;
    --robot|-r) ROBOT="${2:?}"; shift 2 ;;
    --out) OUT="${2:?}"; shift 2 ;;
    --install) INSTALL=1; shift ;;
    --verify) VERIFY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage 0 ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
done

[[ -n "$SOURCE" ]] || die "--source is required (train|ufo|partymimic|mimiclite)"
[[ -n "$RUN" || "$SOURCE" == "mimiclite" ]] || die "--run is required"

ROBOT_DIR="$DEPLOY_DIR/src/inference/robots/$ROBOT"
MODELS_DIR="$ROBOT_DIR/models"
RESULT=""

newest_checkpoint() {
  find "$1" -maxdepth 2 -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null \
    | sort -n | tail -1 | cut -d' ' -f2-
}

onnx_info() {
  local file="$1"
  if python3 -c 'import onnx' >/dev/null 2>&1; then
    python3 "$(dirname "${BASH_SOURCE[0]}")/lib/onnx_info.py" "$file" --dof 23
  elif have uv; then
    run uv run --quiet --with onnx python "$(dirname "${BASH_SOURCE[0]}")/lib/onnx_info.py" "$file" --dof 23
  else
    warn "cannot inspect $file: install onnx or uv"
    return 1
  fi
}

export_train() {
  local run="$1" ckpt="$2" task="$3"
  [[ -n "$task" ]] || die "--task is required for --source train (e.g. RPO-AMP-Play)"
  ckpt="${ckpt:-$(newest_checkpoint "$run")}"
  [[ -n "$ckpt" ]] || die "no model_*.pt under $run (pass --checkpoint)"
  log "train: task=$task checkpoint=$(rel "$ckpt")"
  run_in "$TRAIN_DIR" bash -lc "source .venv/bin/activate && python robolab/scripts/rsl_rl/play.py \
    --task='$task' --checkpoint='$ckpt' --num_envs=1 --viz=none"
  RESULT="$(dirname "$ckpt")/exported/policy.onnx"
}

export_ufo() {
  local run="$1"
  require_dir "$run"
  log "ufo: tracking inference with --export-onnx (writes next to the checkpoint)"
  run_in "$MOTION_DIR/UFO" ./run_inference.sh tracking \
    --model-folder "$run" --motion-list "$MOTION_LIST" --export-onnx
  RESULT="$(find "$run" -maxdepth 3 -name '*.onnx' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
}

export_partymimic() {
  local ckpt="$1"
  require_file "$ckpt"
  local out="${OUT:-$(dirname "$ckpt")/exported/policy_${NAME:-partymimic}.onnx}"
  run mkdir -p "$(dirname "$out")"
  log "partymimic: export_deploy_onnx"
  run_in "$MOTION_DIR/PartyMimic" uv run python tools/export_deploy_onnx.py "$ckpt" "$out"
  RESULT="$out"
}

export_mimiclite() {
  local task="${TASK:-tracking-base}"
  log "mimiclite: play.py with export_policy=true"
  run_in "$MOTION_DIR/MimicLite/active-adaptation" bash -lc "uv run --project venv/mjlab \
    projects/mimic-lite/scripts/play.py task='$task' task/robot='$ROBOT_PROFILE' \
    checkpoint_path='${CHECKPOINT:-${RUN}}' export_policy=true ++export_only=true"
  RESULT="$(find "${RUN:-$MOTION_DIR/MimicLite}" -maxdepth 4 -name '*.onnx' -printf '%T@ %p\n' \
    2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)"
}

install_policy() {
  local file="$1"
  local name="${NAME:-$(basename "$file" .onnx)}"
  require_file "$file"
  run mkdir -p "$MODELS_DIR"
  run cp -f "$file" "$MODELS_DIR/policy_${name}.onnx"
  ok "installed models/policy_${name}.onnx"

  # Sidecars: external weights, metadata, deploy yaml.
  local side
  for side in "$file".data "${file%.onnx}.meta.json" "${file%.onnx}.deploy.yaml"; do
    [[ -f "$side" ]] && run cp -f "$side" "$MODELS_DIR/"
  done
  # External-weight files are named <model>.onnx.data; also catch any *.onnx.data sibling.
  while IFS= read -r side; do
    run cp -f "$side" "$MODELS_DIR/"
  done < <(find "$(dirname "$file")" -maxdepth 1 -name '*.onnx.data' 2>/dev/null)

  if [[ "$SOURCE" == "ufo" ]]; then
    local lat
    lat="$(find "$(dirname "$file")" -maxdepth 2 \( -name '*_z.npz' -o -name '*latent*.npz' -o -name 'z*.npz' \) 2>/dev/null | head -1)"
    if [[ -n "$lat" ]]; then
      run mkdir -p "$ROBOT_DIR/latents"
      run cp -f "$lat" "$ROBOT_DIR/latents/"
      ok "installed latents/$(basename "$lat")"
    else
      warn "no latent .npz found next to $file — configs/ufo.yaml needs latent_names to resolve"
    fi
  fi

  cat <<EOF

$(ok "deploy wiring")
  policy YAML must reference this file, e.g.:
    $(rel "$MODELS_DIR")/policy_${name}.onnx
  in $([ -f "$ROBOT_DIR/configs/$name.yaml" ] && echo "configs/$name.yaml (exists)" || echo "configs/${name}.yaml (create it from an existing policy and update model_names, obs_layouts, usd2urdf)")
  then start it with:
    $(rel "$NYRA_ROOT")/05_integration/scripts/launch_base.sh --robot $ROBOT --policy ${name}
EOF
}

main() {
  log "workspace: $NYRA_ROOT"
  (( DRY_RUN )) && warn "dry run — nothing will be changed"

  case "$SOURCE" in
    train)      export_train "$RUN" "$CHECKPOINT" "$TASK" ;;
    ufo)        export_ufo "$RUN" ;;
    partymimic) export_partymimic "${CHECKPOINT:-$RUN}" ;;
    mimiclite)  export_mimiclite ;;
    *) die "unknown --source '$SOURCE' (train|ufo|partymimic|mimiclite)" ;;
  esac

  if (( DRY_RUN )); then
    dim "dry run: exported artefact would be ${RESULT:-<unknown>}"
    exit 0
  fi

  [[ -n "$RESULT" && -f "$RESULT" ]] || die "export did not produce an ONNX file (looked for: ${RESULT:-<none>})"
  ok "exported $(rel "$RESULT") ($(du -h "$RESULT" | cut -f1))"

  (( VERIFY )) && { step "ONNX contract"; onnx_info "$RESULT" || warn "verification skipped"; }

  if (( INSTALL )); then
    step "Install into roboparty_deploy"
    install_policy "$RESULT"
  fi
}

main
