#!/usr/bin/env bash
# retarget_and_train.sh — motion in, trained policy out.
#
#   ./retarget_and_train.sh --input data/walk.bvh --target train --motion-format amp
#   ./retarget_and_train.sh --input data/clips/ --target ufo --train
#   ./retarget_and_train.sh --input data/walk.npz --target mimiclite
#   ./retarget_and_train.sh --input data/walk.npz --target partymimic --trainer-robot rp1
#
# Stages: (1) hhtools robot preset, (2) retarget to CSV, (3) convert for the target,
#         (4) optionally launch training.
#
#   --input PATH            motion file (.npz/.pkl/.bvh/.glb) or a directory
#   --target train|ufo|mimiclite|partymimic     (default train)
#   --robot-preset NAME     hhtools robot preset (default roboto_origin)
#   --urdf PATH             register/verify this URDF as the preset (default rpo_description)
#   --trainer-robot rpo|rp1 robot selector for roboparty_train (default rpo)
#   --motion-format amp|bm  roboparty_train dataset format (default amp)
#   --name NAME             clip name (default: input stem)
#   --out DIR               working directory (default 06_assets/motions/<name>)
#   --limit-frames N        smoke test: retarget only N frames
#   --calibration-reference NAME   hhtools calibration preset (e.g. smpl), if needed
#   --train                 launch training after conversion
#   --force                 continue past non-fatal incompatibilities
#   --dry-run               print commands only

set -uo pipefail
# shellcheck source=lib/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

INPUT=""
TARGET=train
PRESET=roboto_origin
URDF="$DESCRIPTION_DIR/urdf/rpo.urdf"
TRAINER_ROBOT=rpo
MOTION_FORMAT=amp
NAME=""
OUT=""
LIMIT_FRAMES=""
CALIB=""
DO_TRAIN=0
FORCE=0

usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"; exit "${1:-0}"; }

while (($#)); do
  case "$1" in
    --input|-i) INPUT="${2:?}"; shift 2 ;;
    --target) TARGET="${2:?}"; shift 2 ;;
    --robot-preset) PRESET="${2:?}"; shift 2 ;;
    --urdf) URDF="${2:?}"; shift 2 ;;
    --trainer-robot) TRAINER_ROBOT="${2:?}"; shift 2 ;;
    --motion-format) MOTION_FORMAT="${2:?}"; shift 2 ;;
    --name) NAME="${2:?}"; shift 2 ;;
    --out) OUT="${2:?}"; shift 2 ;;
    --limit-frames) LIMIT_FRAMES="${2:?}"; shift 2 ;;
    --calibration-reference) CALIB="${2:?}"; shift 2 ;;
    --train) DO_TRAIN=1; shift ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --yes|-y) ASSUME_YES=1; shift ;;
    -h|--help) usage 0 ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
done

[[ -n "$INPUT" ]] || die "--input is required"
[[ -e "$INPUT" ]] || die "input not found: $INPUT"
[[ -n "$NAME" ]] || NAME="$(basename "$INPUT")"; NAME="${NAME%.*}"; NAME="${NAME%.npz}"
[[ -n "$OUT" ]] || OUT="$ASSETS_DIR/motions/$NAME"

HHT="$MOTION_DIR/human-humanoid-tools"
CSV_DIR="$OUT/csv"
CONVERT="$NYRA_ROOT/05_integration/scripts/lib/hhtools_csv_convert.py"
PY=python3

# --------------------------------------------------------------------------- 1. robot preset

ensure_preset() {
  step "1/4 hhtools robot preset: $PRESET"
  require_dir "$HHT"
  if (( DRY_RUN )); then
    dim "would check: uv run hhtools robot list | grep $PRESET"
    return 0
  fi
  local list
  list="$( cd "$HHT" && uv run hhtools robot list 2>/dev/null )" || die "hhtools not installed (run setup_all.sh --motion)"

  if grep -qE "(^|[[:space:]])$PRESET([[:space:]]|$)" <<<"$list"; then
    ok "preset '$PRESET' available"
    ( cd "$HHT" && uv run hhtools robot info "$PRESET" --no-mjcf 2>/dev/null | head -12 ) || true
  else
    warn "preset '$PRESET' not found in hhtools"
    if [[ -f "$URDF" ]]; then
      log "registering $(rel "$URDF") as '$PRESET'"
      run_in "$HHT" uv run hhtools robot add "$URDF"
      # hhtools derives the preset name from the URDF; keep the user's name stable.
      local derived
      derived="$( cd "$HHT" && uv run hhtools robot list 2>/dev/null | tail -3 | tr -d ' ' )"
      dim "hhtools now lists: $(echo "$derived" | tr '\n' ' ')"
      dim "if the derived name differs, pass --robot-preset <that name>"
    else
      die "no preset '$PRESET' and no URDF at $(rel "$URDF")"
    fi
  fi
}

# --------------------------------------------------------------------------- 2. retarget

do_retarget() {
  step "2/4 retarget → CSV"
  run mkdir -p "$CSV_DIR"
  local args=(uv run hhtools retarget run "$INPUT" --robot "$PRESET" -o "$CSV_DIR")
  [[ -n "$CALIB" ]] && args+=(--calibration-reference "$CALIB")
  [[ -n "$LIMIT_FRAMES" ]] && args+=(--limit-frames "$LIMIT_FRAMES")
  run_in "$HHT" "${args[@]}"
  if ! (( DRY_RUN )); then
    local n
    n="$(find "$CSV_DIR" -name '*.csv' | wc -l)"
    (( n > 0 )) || die "no CSV produced in $(rel "$CSV_DIR")"
    ok "$n CSV clip(s) in $(rel "$CSV_DIR")"
  fi
}

train_task_id() {
  case "$MOTION_FORMAT" in
    amp) printf '%s' "RPO-AMP" ;;
    bm)  printf '%s' "RPO-BeyondMimic" ;;
  esac
}

first_csv() {
  local f
  f="$(find "$CSV_DIR" -name '*.csv' 2>/dev/null | sort | head -1)"
  if [[ -n "$f" ]]; then
    printf '%s' "$f"
  elif (( DRY_RUN )); then
    printf '%s' "<retargeted.csv>"
  else
    die "no CSV under $(rel "$CSV_DIR")"
  fi
}

# --------------------------------------------------------------------------- 3. convert

convert_train() {
  step "3/4 convert for roboparty_train ($TRAINER_ROBOT, $MOTION_FORMAT)"
  local csv fps dst ext
  csv="$(first_csv)"
  if [[ "$MOTION_FORMAT" == "amp" ]]; then fps=120; dst="robolab/data/motions/${TRAINER_ROBOT}_amp"; ext=pkl
  else fps=50; dst="robolab/data/motions/${TRAINER_ROBOT}_bm"; ext=npz; fi
  run mkdir -p "$TRAIN_DIR/$dst"
  run_in "$TRAIN_DIR" bash -lc "source .venv/bin/activate && python robolab/scripts/tools/hhtools/convert_motion.py \
    --robot '$TRAINER_ROBOT' --output_fps '$fps' -f '$csv' -o '$dst/$NAME.$ext'"
  ok "dataset target: $(rel "$TRAIN_DIR/$dst")/$NAME.$ext"
  dim "train with: python robolab/scripts/rsl_rl/train.py --task=$(train_task_id) ..."
}

convert_ufo() {
  step "3/4 convert for UFO (robot_state_csv)"
  require_dir "$MOTION_DIR/UFO"
  local csv out
  csv="$(first_csv)"
  out="$CSV_DIR/ufo/$NAME.csv"
  run mkdir -p "$CSV_DIR/ufo"
  run "$PY" "$CONVERT" --target ufo -i "$csv" -o "$out" --urdf "$URDF"

  local cfg="$MOTION_DIR/UFO/configs/robots/${TRAINER_ROBOT}.yaml"
  if [[ -f "$cfg" ]]; then
    ok "robot config shipped by UFO: $(rel "$cfg")"
  else
    log "generating UFO robot config from $(rel "$URDF")"
    run_in "$MOTION_DIR/UFO" uv run python -m ufo.tools.robot_setup \
      --name "$TRAINER_ROBOT" --xml "$DESCRIPTION_DIR/mjcf/rpo.xml" --urdf "$URDF" \
      --output "$cfg" --motion-source "$CSV_DIR/ufo/*.csv" --fps 50
    warn "review the generated YAML (gains, action scale, body groups) before a long run"
  fi
  ok "motion source: $(rel "$CSV_DIR/ufo")"
  dim "train with: cd $(rel "$MOTION_DIR/UFO") && ./run_train.sh --robot $TRAINER_ROBOT --agent fb --gpu-ids single --smoke"
}

convert_mimiclite() {
  step "3/4 convert for MimicLite (any4hdmi qpos)"
  local any4="$MOTION_DIR/any4hdmi"
  require_dir "$any4"
  local csv
  csv="$(first_csv)"
  run mkdir -p "$CSV_DIR/any4hdmi"
  run "$PY" "$CONVERT" --target any4hdmi -i "$csv" -o "$CSV_DIR/any4hdmi/$NAME.csv" --urdf "$URDF"

  # any4hdmi's converters hardcode a 29-joint G1 order; RPO needs a one-constant change.
  local expected
  expected="$($PY - "$any4/src/any4hdmi/core/model.py" <<'PY'
import re, sys, pathlib
src = pathlib.Path(sys.argv[1]).read_text()
m = re.search(r"G1_JOINT_ORDER\s*=\s*\[(.*?)\]", src, re.S)
print(len(re.findall(r'"', m.group(1))) // 2 if m else -1)
PY
)"
  local target_joints
  target_joints="$($PY - "$URDF" <<'PY'
import sys, xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
print(sum(1 for j in root.findall("joint") if j.get("type") != "fixed"))
PY
)"
  dim "any4hdmi expects $expected joints, $URDF has $target_joints"

  if [[ "$expected" != "$target_joints" ]]; then
    warn "any4hdmi hardcodes a ${expected}-joint G1 list; the target has ${target_joints} joints."
    dim "patch src/any4hdmi/core/model.py: G1_JOINT_ORDER → the ${target_joints} joints of $(basename "$URDF")"
    dim "then: uv run any4hdmi-convert-lafan --csv-dir '$CSV_DIR/any4hdmi' --out-dir '<mimic-lite>/../any4hdmi/output/atom/sonic' \\"
    dim "        --mjcf-repo elijahgalahad/atom01 --mjcf-path mjcf/atom01-capsule.xml --fps 50"
    (( FORCE )) || { fail "stopping: patch any4hdmi or pass --force to continue anyway"; return 1; }
  else
    run_in "$any4" uv run any4hdmi-convert-lafan --csv-dir "$CSV_DIR/any4hdmi" \
      --out-dir "$MOTION_DIR/MimicLite/any4hdmi/output/atom/sonic" \
      --mjcf-repo elijahgalahad/atom01 --mjcf-path mjcf/atom01-capsule.xml --fps 50
  fi
  dim "train with: task=tracking-base task/robot=atom (see README §6)"
}

convert_partymimic() {
  step "3/4 convert for PartyMimic (qpos npz)"
  local pm="$MOTION_DIR/PartyMimic"
  require_dir "$pm"
  if [[ ! -d "$pm/src/partymimic/$TRAINER_ROBOT" ]]; then
    local pkgs=()
    local entry
    for entry in "$pm/src/partymimic"/*/; do pkgs+=("$(basename "$entry")"); done
    warn "PartyMimic has no robot package '$TRAINER_ROBOT' (only: ${pkgs[*]})"
    dim "add src/partymimic/$TRAINER_ROBOT/__init__.py + ${TRAINER_ROBOT}_constants.py, mirroring rp1:"
    dim "  MJCF_PATH, MOTION_DIR, ACTUATORS (kp/kd), DEFAULT_JOINT_POS, TRACKED_BODIES, RobotSpec"
    dim "  kp/kd and default angles: $(rel "$DEPLOY_DIR")/src/inference/robots/rpo/configs/default.yaml"
    (( FORCE )) || { fail "stopping: no PartyMimic robot package for '$TRAINER_ROBOT'"; return 1; }
  fi
  local csv
  csv="$(first_csv)"
  run mkdir -p "$CSV_DIR/any4hdmi"
  run "$PY" "$CONVERT" --target any4hdmi -i "$csv" -o "$CSV_DIR/any4hdmi/$NAME.csv" --urdf "$URDF"
  dim "PartyMimic consumes any4hdmi qpos npz: convert with any4hdmi, then"
  dim "  uv run python tools/data/prep_motions.py --format qpos --src <qpos dir> --out data/motions/${TRAINER_ROBOT}_mix"
  dim "  bash scripts/train.sh logs/${TRAINER_ROBOT}_s0 --robot $TRAINER_ROBOT --motion_dir data/motions/${TRAINER_ROBOT}_mix"
}

# --------------------------------------------------------------------------- 4. train

train_now() {
  step "4/4 launch training ($TARGET)"
  case "$TARGET" in
    train)
      run_in "$TRAIN_DIR" bash -lc "source .venv/bin/activate && python robolab/scripts/rsl_rl/train.py \
        --task='$(train_task_id)' --viz=none --logger=tensorboard --num_envs=4096"
      ;;
    ufo)
      run_in "$MOTION_DIR/UFO" ./run_train.sh --robot "$TRAINER_ROBOT" --agent fb --gpu-ids all \
        --work-dir "runs/${TRAINER_ROBOT}_fb_${NAME}"
      ;;
    mimiclite)
      run_in "$MOTION_DIR/MimicLite/active-adaptation" bash scripts/launch_ddp.sh 0 \
        projects/mimic-lite/scripts/train.py venv/mjlab \
        task=tracking-base task/robot=atom backend=mjlab +exp=ppo/train wandb.mode=disabled
      ;;
    partymimic)
      run_in "$MOTION_DIR/PartyMimic" bash scripts/train.sh "logs/${TRAINER_ROBOT}_${NAME}" \
        --robot "$TRAINER_ROBOT" --motion_dir "data/motions/${TRAINER_ROBOT}_mix" --seed 0
      ;;
  esac
}

main() {
  log "workspace: $NYRA_ROOT"
  log "input:     $INPUT"
  log "target:    $TARGET ($TRAINER_ROBOT)"
  log "work dir:  $(rel "$OUT")"
  (( DRY_RUN )) && warn "dry run — nothing will be changed"

  case "$TARGET" in
    train|ufo|mimiclite|partymimic) ;;
    *) die "unknown --target '$TARGET' (train|ufo|mimiclite|partymimic)" ;;
  esac

  ensure_preset
  do_retarget
  "convert_$TARGET" || { fail "conversion failed"; exit 1; }

  if (( DO_TRAIN )); then
    (( ASSUME_YES )) || confirm "Start training now?" || { log "stopped after conversion"; exit 0; }
    train_now
  else
    step "done (conversion only)"
    dim "re-run with --train to launch training"
  fi

  cat <<EOF

$(ok "pipeline finished")
  next:
    05_integration/scripts/export_policy.sh --source $([[ $TARGET == ufo ]] && echo ufo || echo train) \\
      --run <run dir> --name $NAME --install --verify
EOF
}

main
