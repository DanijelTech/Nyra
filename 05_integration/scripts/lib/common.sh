#!/usr/bin/env bash
# Shared helpers for the Nyra integration scripts. Source, do not execute.

set -uo pipefail

if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'; C_RED=$'\033[31m'; C_GREEN=$'\033[32m'
  C_YELLOW=$'\033[33m'; C_BLUE=$'\033[34m'; C_DIM=$'\033[2m'; C_BOLD=$'\033[1m'
else
  C_RESET=""; C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""; C_DIM=""; C_BOLD=""
fi

NYRA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export NYRA_ROOT


export DEPLOY_DIR="$NYRA_ROOT/00_body_base/roboparty_deploy"
export DESCRIPTION_DIR="$NYRA_ROOT/00_body_base/rpo_description"
export TRAIN_DIR="$NYRA_ROOT/04_training/roboparty_train"
export ISAACLAB_DIR="$NYRA_ROOT/04_training/IsaacLab-3.0.0"
export MOTION_DIR="$NYRA_ROOT/02_motion_infra"
export ADVANCED_DIR="$NYRA_ROOT/03_advanced"
export ASSETS_DIR="$NYRA_ROOT/06_assets"

DRY_RUN=0
ASSUME_YES=0

log()   { printf '%s[nyra]%s %s\n' "$C_BLUE" "$C_RESET" "$*"; }
step()  { printf '\n%s==> %s%s\n' "$C_BOLD" "$*" "$C_RESET"; }
ok()    { printf '%s  ok%s   %s\n' "$C_GREEN" "$C_RESET" "$*"; }
warn()  { printf '%s  warn%s %s\n' "$C_YELLOW" "$C_RESET" "$*" >&2; }
fail()  { printf '%s  fail%s %s\n' "$C_RED" "$C_RESET" "$*" >&2; }
dim()   { printf '%s       %s%s\n' "$C_DIM" "$*" "$C_RESET"; }
die()   { fail "$*"; exit 1; }

# run <cmd...> — echo and execute, honouring --dry-run.
run() {
  if (( DRY_RUN )); then
    printf '%s  dry%s  %s\n' "$C_DIM" "$C_RESET" "$*"
    return 0
  fi
  printf '%s  run%s  %s\n' "$C_DIM" "$C_RESET" "$*"
  "$@"
}

# run_in <dir> <cmd...> — same, but from a different working directory.
run_in() {
  local dir="$1"; shift
  if (( DRY_RUN )); then
    printf '%s  dry%s  (cd %s && %s)\n' "$C_DIM" "$C_RESET" "${dir#"$NYRA_ROOT"/}" "$*"
    return 0
  fi
  printf '%s  run%s  (cd %s && %s)\n' "$C_DIM" "$C_RESET" "${dir#"$NYRA_ROOT"/}" "$*"
  ( cd "$dir" && "$@" )
}

have() { command -v "$1" >/dev/null 2>&1; }

confirm() {
  (( ASSUME_YES )) && return 0
  local answer
  read -r -p "$1 [y/N] " answer
  [[ "$answer" =~ ^[Yy]$ ]]
}

require_dir() {
  [[ -d "$1" ]] || die "missing directory: ${1#"$NYRA_ROOT"/} (run: git submodule update --init --recursive)"
}

require_file() {
  [[ -f "$1" ]] || die "missing file: ${1#"$NYRA_ROOT"/}"
}

ros_setup_file() {
  local distro="${ROS_DISTRO:-}"
  if [[ -n "$distro" && -f "/opt/ros/$distro/setup.bash" ]]; then
    echo "/opt/ros/$distro/setup.bash"; return 0
  fi
  local candidate
  for candidate in /opt/ros/*/setup.bash; do
    [[ -f "$candidate" ]] && { echo "$candidate"; return 0; }
  done
  return 1
}

submodules_ready() {
  local dir="$1" missing
  missing="$(git -C "$dir" submodule status 2>/dev/null | grep -c '^-' || true)"
  [[ "${missing:-0}" -eq 0 ]]
}

# Relative path for readable output.
rel() { printf '%s' "${1#"$NYRA_ROOT"/}"; }
