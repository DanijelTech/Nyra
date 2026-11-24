#!/usr/bin/env bash
# launch_base.sh — start (or stop) the base locomotion stack on the robot.
#
#   ./launch_base.sh                          # robot=rpo, policy=default
#   ./launch_base.sh --policy amp             # walk / run
#   ./launch_base.sh --policy parkour --camera
#   ./launch_base.sh --preflight-only         # run the checks, start nothing
#   ./launch_base.sh --stop                   # quit the screen sessions
#
#   --robot rpo|rp1     robot profile (default rpo)
#   --policy NAME       policy config from src/inference/robots/<robot>/configs/
#   --camera            also start start_camera.sh (needed by parkour)
#   --force             start even if a preflight check fails
#   --dry-run           print commands only

set -uo pipefail
# shellcheck source=lib/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

ROBOT=rpo
POLICY=default
WITH_CAMERA=0
PREFLIGHT_ONLY=0
STOP=0
FORCE=0

usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"; exit "${1:-0}"; }

while (($#)); do
  case "$1" in
    --robot|-r) ROBOT="${2:?--robot needs a value}"; shift 2 ;;
    --policy|-p) POLICY="${2:?--policy needs a value}"; shift 2 ;;
    --camera) WITH_CAMERA=1; shift ;;
    --preflight-only) PREFLIGHT_ONLY=1; shift ;;
    --stop) STOP=1; shift ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --yes|-y) ASSUME_YES=1; shift ;;
    -h|--help) usage 0 ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
done

ROBOT_DIR="$DEPLOY_DIR/src/inference/robots/$ROBOT"
CONFIG_DIR="$ROBOT_DIR/configs"

stop_sessions() {
  step "Stopping screen sessions"
  local s
  for s in inference_session joy_session realsense_session camera_session; do
    if screen -ls 2>/dev/null | grep -q "[.]$s"; then
      run screen -S "$s" -X quit
    else
      dim "no session: $s"
    fi
  done
}

preflight() {
  step "Preflight: robot=$ROBOT policy=$POLICY"
  local problems=()

  if [[ -d "$DEPLOY_DIR" && -n "$(ls -A "$DEPLOY_DIR" 2>/dev/null)" ]]; then
    ok "roboparty_deploy present"
  else
    problems+=("roboparty_deploy is empty — git submodule update --init --recursive")
  fi

  if [[ -f "$DEPLOY_DIR/tools/start_robot.sh" ]]; then
    ok "tools/start_robot.sh"
  else
    problems+=("tools/start_robot.sh missing")
  fi

  if submodules_ready "$DEPLOY_DIR"; then
    ok "deploy submodules initialised (inference, motors, imu, camera)"
  else
    problems+=("deploy submodules not initialised: git -C 00_body_base/roboparty_deploy submodule update --init --recursive")
  fi

  if [[ -f "$CONFIG_DIR/$POLICY.yaml" ]]; then
    ok "policy config: configs/$POLICY.yaml"
  else
    local available
    available="$(ls "$CONFIG_DIR" 2>/dev/null | sed 's/\.yaml$//' | tr '\n' ' ')"
    problems+=("no configs/$POLICY.yaml — available: ${available:-none}")
  fi

  # Model file referenced by the policy YAML must exist.
  if [[ -f "$CONFIG_DIR/$POLICY.yaml" ]]; then
    local model
    model="$(sed -n 's/.*model_names: *\["\([^"]*\)".*/\1/p' "$CONFIG_DIR/$POLICY.yaml" | head -1)"
    if [[ -z "$model" ]]; then
      dim "configs/$POLICY.yaml declares no model_names (check manually)"
    elif [[ -f "$ROBOT_DIR/models/$model" ]]; then
      ok "model: models/$model"
    else
      problems+=("models/$model referenced by configs/$POLICY.yaml is missing (export_policy.sh --install)")
    fi
  fi

  if [[ -f "$ROBOT_DIR/robot.yaml" ]]; then
    ok "hardware config: robot.yaml"
  else
    problems+=("$ROBOT_DIR/robot.yaml missing")
  fi

  local ros
  if ros="$(ros_setup_file)"; then
    ok "ROS 2: $ros"
  else
    problems+=("ROS 2 not installed (source /opt/ros/<distro>/setup.bash)")
  fi

  if have colcon; then ok "colcon"; else problems+=("colcon missing (apt install python3-colcon-common-extensions)"); fi

  # Hardware checks are informational: a dev box legitimately has no CAN bus.
  local cans
  cans="$(ip -br link show 2>/dev/null | awk '/^can/{print $1}' | tr '\n' ' ')"
  if [[ -n "$cans" ]]; then
    ok "CAN interfaces: $cans"
    local c
    for c in $cans; do
      if ip -details link show "$c" 2>/dev/null | grep -q "state UP"; then
        dim "$c is UP"
      else
        warn "$c is DOWN — sudo ip link set $c up type can bitrate 1000000"
      fi
    done
  else
    dim "no CAN interfaces (expected on a dev box; required on the robot)"
  fi

  local rt
  rt="$(ulimit -r 2>/dev/null || echo 0)"
  if [[ "$rt" =~ ^[0-9]+$ ]] && (( rt >= 98 )); then
    ok "rtprio $rt"
  else
    warn "rtprio ${rt:-unknown} < 98 — see docs/rpo_deploy.md (limits.conf + reboot)"
  fi

  if [[ "$POLICY" == "parkour" && $WITH_CAMERA -eq 0 ]]; then
    warn "policy parkour needs the camera node: add --camera"
  fi

  if ((${#problems[@]})); then
    step "Preflight problems"
    local p
    for p in "${problems[@]}"; do
      printf '%s  - %s%s\n' "$C_RED" "$p" "$C_RESET"
    done
    if (( FORCE )); then
      warn "continuing anyway (--force)"
    else
      fail "refusing to start; fix the items above or pass --force"
      return 1
    fi
  else
    ok "preflight clean"
  fi
  return 0
}

start_stack() {
  local ros
  ros="$(ros_setup_file)" || die "ROS 2 not found"

  step "Starting base control ($ROBOT / $POLICY)"
  if (( WITH_CAMERA )); then
    log "camera node first (policy $POLICY reads /depth_obs)"
    run_in "$DEPLOY_DIR" ./tools/start_camera.sh --policy "$POLICY"
    sleep 2
  fi

  # start_robot.sh builds the inference package, stops stale sessions and starts
  # inference_session + joy_session in screen.
  run_in "$DEPLOY_DIR" ./tools/start_robot.sh --robot "$ROBOT" --policy "$POLICY"

  cat <<EOF

$(ok "started")
  attach:  screen -r inference_session      (detach: C-a d)
           screen -r joy_session
           screen -r camera_session          (only with --camera)
  stop:    $(rel "$NYRA_ROOT")/05_integration/scripts/launch_base.sh --stop

$(warn "Safety: zero the motors, and suspend or stand the robot for the first run of any new policy.")
EOF
}

main() {
  log "workspace: $NYRA_ROOT"
  (( DRY_RUN )) && warn "dry run — nothing will be changed"

  if (( STOP )); then
    stop_sessions
    exit 0
  fi

  require_dir "$DEPLOY_DIR"
  preflight || exit 1
  (( PREFLIGHT_ONLY )) && exit 0

  if ! (( ASSUME_YES )); then
    confirm "Start $ROBOT with policy $POLICY now?" || { log "aborted"; exit 0; }
  fi
  start_stack
}

main
