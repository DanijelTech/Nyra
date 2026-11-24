#!/usr/bin/env bash
# setup_all.sh — prepare the Nyra workspace tier by tier.
#
#   ./setup_all.sh --check                 report what is installed / missing
#   ./setup_all.sh --base                  ROS 2 + C++ deps + colcon build of roboparty_deploy
#   ./setup_all.sh --motion                uv envs for hhtools / any4hdmi / PartyMimic / UFO / MimicLite
#   ./setup_all.sh --train                 uv env for roboparty_train (needs Isaac Lab 3.0)
#   ./setup_all.sh --advanced              PredActor / P-BFM / I-BFM / SADL
#   ./setup_all.sh --all                   every phase, in order
#
#   --with-data    also download motion datasets (HF_TOKEN needed, several GB)
#   --physx        create roboparty_train's PhysX environment as well
#   --no-apt       never call apt-get
#   --dry-run      print the commands, change nothing
#   --yes          no confirmation prompts

set -uo pipefail
# shellcheck source=lib/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

PHASES=()
WITH_DATA=0
PHYSX=0
NO_APT=0
CHECK_ONLY=0
FAILED=()
SKIPPED=()

usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"; exit "${1:-0}"; }

while (($#)); do
  case "$1" in
    --base|--motion|--train|--advanced) PHASES+=("${1#--}") ;;
    --all) PHASES=(base motion train advanced) ;;
    --check) CHECK_ONLY=1 ;;
    --with-data) WITH_DATA=1 ;;
    --physx) PHYSX=1 ;;
    --no-apt) NO_APT=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --yes|-y) ASSUME_YES=1 ;;
    -h|--help) usage 0 ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
  shift
done

[[ $CHECK_ONLY -eq 0 && ${#PHASES[@]} -eq 0 ]] && PHASES=(base motion)

record_fail() { FAILED+=("$1"); fail "$1"; }
record_skip() { SKIPPED+=("$1"); warn "$1"; }

# --------------------------------------------------------------------------- helpers

ensure_uv() {
  if have uv; then return 0; fi
  warn "uv not found; installing with pip --user (https://docs.astral.sh/uv/)"
  run python3 -m pip install --user uv || { record_fail "uv install"; return 1; }
  export PATH="$HOME/.local/bin:$PATH"
  have uv
}

apt_install() {
  local pkgs=("$@")
  if (( NO_APT )); then dim "apt-get skipped (--no-apt): ${pkgs[*]}"; return 0; fi
  if ! have apt-get; then warn "apt-get not available; install manually: ${pkgs[*]}"; return 0; fi
  local sudo_cmd=()
  [[ $EUID -ne 0 ]] && sudo_cmd=(sudo)
  (( ASSUME_YES )) && export DEBIAN_FRONTEND=noninteractive
  run "${sudo_cmd[@]}" apt-get update -qq || true
  run "${sudo_cmd[@]}" apt-get install -y "${pkgs[@]}" || record_fail "apt install ${pkgs[*]}"
}

# --------------------------------------------------------------------------- phases

phase_submodules() {
  step "Submodules"
  require_dir "$NYRA_ROOT/.git"
  run git -C "$NYRA_ROOT" submodule sync --recursive
  run git -C "$NYRA_ROOT" submodule update --init --recursive
  # MimicLite ships SSH URLs; map them to HTTPS for keyless clones.
  if [[ -d "$MOTION_DIR/MimicLite/.git" || -f "$MOTION_DIR/MimicLite/.git" ]]; then
    run git -C "$MOTION_DIR/MimicLite" config url."https://github.com/".insteadOf "git@github.com:"
    run git -C "$MOTION_DIR/MimicLite" submodule update --init --recursive
  fi
  local empty=()
  local pair
  for pair in "$DEPLOY_DIR/src/inference" "$TRAIN_DIR/robolab" "$TRAIN_DIR/rsl_rl" "$MOTION_DIR/MimicLite/mimic-lite"; do
    [[ -d "$pair" && -n "$(ls -A "$pair" 2>/dev/null)" ]] || empty+=("$(rel "$pair")")
  done
  if ((${#empty[@]})); then
    record_fail "nested submodules still empty: ${empty[*]}"
  else
    ok "nested submodules populated"
  fi
}

phase_base() {
  step "Tier 0 — base control (ROS 2 + roboparty_deploy)"
  apt_install ccache libfmt-dev libspdlog-dev libeigen3-dev screen zlib1g-dev
  apt_install python3-yaml python3-numpy

  local ros
  if ros="$(ros_setup_file)"; then
    ok "ROS 2 found: $ros"
    local distro="${ros#/opt/ros/}"; distro="${distro%/setup.bash}"
    apt_install "ros-${distro}-joy"
  else
    record_skip "ROS 2 not installed — install Humble or Jazzy before building roboparty_deploy"
  fi

  if ! have colcon; then
    record_skip "colcon not installed (apt install python3-colcon-common-extensions)"
  else
    run_in "$DEPLOY_DIR" colcon build --symlink-install || record_fail "colcon build (roboparty_deploy)"
  fi

  cat <<EOF

${C_YELLOW}Manual, one-time system steps (not automated on purpose):${C_RESET}
  1) Real-time priority — append to /etc/security/limits.conf, then reboot:
       $(whoami)   -   rtprio   98
       $(whoami)   -   memlock  unlimited
     verify: ulimit -r    # expect 98
  2) udev rules for CAN + IMU:
       sudo cp $(rel "$DEPLOY_DIR")/assets/99-auto-up-devs-orangepi.rules /etc/udev/rules.d/
       sudo udevadm control --reload && sudo udevadm trigger
  3) Motor zeroing + safety guide:
       $(rel "$DEPLOY_DIR")/docs/rpo_deploy.md
EOF
}

phase_motion() {
  step "Tier 1 — motion infra"
  ensure_uv || return 1

  # hhtools: retargeting core
  if [[ -f "$MOTION_DIR/human-humanoid-tools/pyproject.toml" ]]; then
    run_in "$MOTION_DIR/human-humanoid-tools" uv sync --locked --extra all \
      || record_fail "uv sync hhtools"
    run_in "$MOTION_DIR/human-humanoid-tools" uv run python scripts/install_builtin_robots.py \
      || record_fail "hhtools install_builtin_robots"
  else
    record_skip "human-humanoid-tools missing"
  fi

  # any4hdmi: motion format + converters used by MimicLite
  if [[ -f "$MOTION_DIR/any4hdmi/pyproject.toml" ]]; then
    run_in "$MOTION_DIR/any4hdmi" uv sync --locked || record_fail "uv sync any4hdmi"
  else
    record_skip "any4hdmi missing"
  fi

  # PartyMimic: model-based tracker
  if [[ -f "$MOTION_DIR/PartyMimic/pyproject.toml" ]]; then
    run_in "$MOTION_DIR/PartyMimic" uv sync || record_fail "uv sync PartyMimic"
    if (( WITH_DATA )); then
      run_in "$MOTION_DIR/PartyMimic" tools/download_data.sh || record_fail "PartyMimic data"
    else
      dim "skipped PartyMimic data download (--with-data to fetch G1/RP1 assets + LAFAN clips)"
    fi
  else
    record_skip "PartyMimic missing"
  fi

  # UFO: unsupervised RL
  if [[ -f "$MOTION_DIR/UFO/run_train.sh" ]]; then
    run_in "$MOTION_DIR/UFO" uv sync --locked || record_fail "uv sync UFO"
    if (( WITH_DATA )); then
      if [[ -n "${HF_TOKEN:-}" ]]; then
        run_in "$MOTION_DIR/UFO" bash scripts/download_data.sh rpo_lafan || record_fail "UFO rpo_lafan data"
      else
        record_skip "UFO data needs HF_TOKEN (export HF_TOKEN=... then rerun --with-data)"
      fi
    else
      dim "skipped UFO data download (--with-data to fetch rpo_lafan)"
    fi
  else
    record_skip "UFO missing"
  fi

  # MimicLite: two uv environments (mjlab, isaaclab).
  # Upstream expects mimic-lite checked out at active-adaptation/projects/mimic-lite;
  # here both are siblings, so link it into place first.
  local ml="$MOTION_DIR/MimicLite/active-adaptation"
  local ml_proj="$ml/projects/mimic-lite"
  if [[ -d "$ml" ]]; then
    if [[ ! -e "$ml_proj" ]]; then
      if [[ -d "$MOTION_DIR/MimicLite/mimic-lite" ]]; then
        run mkdir -p "$ml/projects"
        run ln -s ../../mimic-lite "$ml_proj"
        dim "linked projects/mimic-lite -> ../../mimic-lite (vendored sibling checkout)"
      else
        record_skip "MimicLite submodules not initialised (git -C 02_motion_infra/MimicLite submodule update --init --recursive)"
      fi
    fi
    if [[ -f "$ml_proj/pyproject-mjlab.toml" ]]; then
      run mkdir -p "$ml/venv/mjlab"
      run cp "$ml_proj/pyproject-mjlab.toml" "$ml/venv/mjlab/pyproject.toml"
      run_in "$ml" uv sync --project venv/mjlab || record_fail "uv sync MimicLite (mjlab)"
      run_in "$ml" uv --project venv/mjlab run aa-discover-projects || true
      run_in "$ml" uv --project venv/mjlab run aa-project enable mimic_lite || true
    fi
    if [[ -f "$ml_proj/pyproject-isaaclab.toml" ]]; then
      run mkdir -p "$ml/venv/isaaclab"
      run cp "$ml_proj/pyproject-isaaclab.toml" "$ml/venv/isaaclab/pyproject.toml"
      run_in "$ml" uv sync --project venv/isaaclab || record_skip "uv sync MimicLite (isaaclab) failed — needs Isaac Lab"
    fi
  else
    record_skip "MimicLite missing"
  fi
}

phase_train() {
  step "Tier 1b — roboparty_train (Isaac Lab)"
  ensure_uv || return 1
  if [[ ! -d "$ISAACLAB_DIR" ]]; then
    record_skip "Isaac Lab 3.0.0 not found at $(rel "$ISAACLAB_DIR")"
    dim "clone it with:"
    dim "  git clone --branch v3.0.0 --recursive https://github.com/isaac-sim/IsaacLab.git $(rel "$ISAACLAB_DIR")"
    return 1
  fi
  ok "Isaac Lab present: $(rel "$ISAACLAB_DIR")"
  run_in "$TRAIN_DIR" uv sync --locked || { record_fail "uv sync roboparty_train (MJWarp)"; return 1; }
  if (( PHYSX )); then
    run_in "$TRAIN_DIR" env UV_PROJECT_ENVIRONMENT=.venv-physx \
      uv sync --locked --no-default-groups --extra physx --no-install-package usd-exchange \
      || record_fail "uv sync roboparty_train (PhysX)"
  fi
  run_in "$TRAIN_DIR" bash -c 'source .venv/bin/activate && isaaclab list_envs --keyword RP' \
    || record_skip "isaaclab list_envs failed — check the environment"
}

phase_advanced() {
  step "Tier 2 — advanced layers"
  ensure_uv || return 1

  if [[ -f "$ADVANCED_DIR/PredActor/pyproject.toml" ]]; then
    run_in "$ADVANCED_DIR/PredActor" uv sync --locked || record_fail "uv sync PredActor"
  else
    record_skip "PredActor missing"
  fi

  if [[ -f "$ADVANCED_DIR/P-BFM/pyproject.toml" ]]; then
    run_in "$ADVANCED_DIR/P-BFM" uv sync --extra test || record_fail "uv sync P-BFM"
  else
    record_skip "P-BFM missing"
  fi

  if [[ -f "$ADVANCED_DIR/I-BFM-Code/restore_checkpoint.sh" ]]; then
    if have git-lfs; then
      run_in "$ADVANCED_DIR/I-BFM-Code" git lfs install || true
      run_in "$ADVANCED_DIR/I-BFM-Code" git lfs pull || record_fail "I-BFM git lfs pull"
      run_in "$ADVANCED_DIR/I-BFM-Code" bash restore_checkpoint.sh || record_fail "I-BFM restore_checkpoint"
      run_in "$ADVANCED_DIR/I-BFM-Code/ibfm" uv sync || record_fail "uv sync I-BFM"
    else
      record_skip "git-lfs not installed (apt install git-lfs) — I-BFM checkpoints skipped"
    fi
  else
    record_skip "I-BFM-Code missing"
  fi

  if [[ -f "$ADVANCED_DIR/SADL/requirements.txt" ]]; then
    run python3 -m venv "$ADVANCED_DIR/SADL/.venv" || record_fail "SADL venv"
    run_in "$ADVANCED_DIR/SADL" .venv/bin/python -m pip install -q --upgrade pip
    run_in "$ADVANCED_DIR/SADL" .venv/bin/python -m pip install -q -r requirements.txt \
      || record_fail "SADL requirements (needs CUDA torch)"
  else
    record_skip "SADL missing"
  fi

  if [[ -f "$ADVANCED_DIR/INTACT-JEPA/README.md" ]]; then
    record_skip "INTACT-JEPA is documentation only (training code not released yet)"
  fi
}

# --------------------------------------------------------------------------- --check

check_report() {
  step "Environment report"

  status() { # label test-cmd...
    local label="$1"; shift
    if "$@" >/dev/null 2>&1; then ok "$label"; else warn "$label — missing"; fi
  }

  if ros_setup_file >/dev/null 2>&1; then ok "ROS 2 ($(ros_setup_file))"; else warn "ROS 2 — not installed"; fi
  status "colcon"                    have colcon
  status "uv"                        have uv
  status "nvidia-smi"                have nvidia-smi
  status "ffmpeg"                    have ffmpeg
  status "git-lfs"                   have git-lfs
  status "HF_TOKEN"                  bash -c '[[ -n "${HF_TOKEN:-}" ]]'
  status "Isaac Lab 3.0.0"           test -d "$ISAACLAB_DIR"

  step "Modules"
  local m
  for m in "$DEPLOY_DIR" "$DESCRIPTION_DIR" "$TRAIN_DIR/robolab" "$MOTION_DIR/human-humanoid-tools" \
           "$MOTION_DIR/any4hdmi" "$MOTION_DIR/PartyMimic" "$MOTION_DIR/UFO" \
           "$MOTION_DIR/MimicLite/mimic-lite" "$ADVANCED_DIR/PredActor" "$ADVANCED_DIR/I-BFM-Code" \
           "$ADVANCED_DIR/P-BFM" "$ADVANCED_DIR/SADL"; do
    if [[ -d "$m" && -n "$(ls -A "$m" 2>/dev/null)" ]]; then ok "$(rel "$m")"; else warn "$(rel "$m") — missing/empty"; fi
  done

  step "Environments"
  local env
  for env in "$MOTION_DIR/human-humanoid-tools/.venv" "$MOTION_DIR/any4hdmi/.venv" \
             "$MOTION_DIR/PartyMimic/.venv" "$MOTION_DIR/UFO/.venv" \
             "$MOTION_DIR/MimicLite/active-adaptation/venv/mjlab/.venv" \
             "$TRAIN_DIR/.venv" "$TRAIN_DIR/.venv-physx" \
             "$ADVANCED_DIR/PredActor/.venv" "$ADVANCED_DIR/P-BFM/.venv" \
             "$ADVANCED_DIR/I-BFM-Code/ibfm/.venv" "$ADVANCED_DIR/SADL/.venv"; do
    if [[ -d "$env" ]]; then ok "$(rel "$env")"; else dim "not created: $(rel "$env")"; fi
  done

  step "Deploy build + policies"
  if [[ -f "$DEPLOY_DIR/install/setup.bash" ]]; then ok "colcon install/ present"; else dim "not built: colcon build --symlink-install"; fi
  local models="$DEPLOY_DIR/src/inference/robots/rpo/models"
  if [[ -d "$models" ]]; then
    ok "models: $(ls "$models" | tr '\n' ' ')"
  else
    warn "no models directory: $models"
  fi

  step "Buses (robot only)"
  if have ip; then
    local cans
    cans="$(ip -br link show 2>/dev/null | awk '/can/{print $1}' | tr '\n' ' ')"
    [[ -n "$cans" ]] && ok "CAN interfaces: $cans" || dim "no CAN interfaces (expected off-robot)"
  fi
  dim "rtprio: $(ulimit -r 2>/dev/null || echo unknown) (expect 98 on the robot)"

  if have python3; then
    step "Compatibility"
    run python3 "$NYRA_ROOT/05_integration/scripts/check_compat.py" --quiet || warn "compat check reported issues"
  fi
}

# --------------------------------------------------------------------------- main

main() {
  log "workspace: $NYRA_ROOT"
  (( DRY_RUN )) && warn "dry run — nothing will be changed"

  if (( CHECK_ONLY )); then
    check_report
    exit 0
  fi

  phase_submodules

  local phase
  for phase in "${PHASES[@]}"; do
    "phase_$phase" || true
  done

  step "Summary"
  if ((${#FAILED[@]})); then
    fail "${#FAILED[@]} step(s) failed:"
    printf '       - %s\n' "${FAILED[@]}"
  else
    ok "all requested steps completed"
  fi
  if ((${#SKIPPED[@]})); then
    dim "skipped / needs attention:"
    printf '       - %s\n' "${SKIPPED[@]}"
  fi
  printf '\nNext: %s/05_integration/scripts/check_compat.py\n' "$NYRA_ROOT"
  ((${#FAILED[@]})) && exit 1
  exit 0
}

main
