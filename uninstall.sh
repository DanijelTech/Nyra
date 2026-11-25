#!/usr/bin/env bash
# uninstall.sh — remove everything THIS project put on this machine.
#
# Default is a DRY RUN: it prints what it would delete (with sizes) and changes
# nothing. Nothing is deleted without --yes.
#
#   ./uninstall.sh                          # dry run: show the full plan
#   ./uninstall.sh --yes --all              # repo + build outputs + envs + npm + uv cache
#   ./uninstall.sh --yes --envs --build     # keep the checkout, free the disk
#   ./uninstall.sh --yes --all --apt        # also remove the ROS 2 packages we installed
#
# Categories (each opt-in; --all means repo+build+envs+npm+uv-cache):
#   --build   build outputs, caches, generated manifests inside the checkout
#   --envs    Python environments this project created (uv venvs, ~/.venvs/nyra)
#   --npm     node_modules/package-lock and the global nyra-workspace package
#   --uv-cache ~/.cache/uv (shared with other uv projects — read the warning)
#   --apt     system packages we told you to install (ROS 2 Jazzy, colcon, ...)
#   --repo    the checkout itself (default with --all; skip with --keep-repo)
#
# It never touches: your git config, the uv/python/node binaries, other projects,
# ~/.local/bin, or anything outside the paths listed below.

set -uo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
# The re-exec'd copy lives in /tmp, so the original checkout travels in the env.
REPO_DEFAULT="${NYRA_UNINSTALL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# If the checkout will be deleted, run from a copy: bash reads its script
# incrementally, so deleting the file underneath it can truncate the run.
if [[ -z "${NYRA_UNINSTALL_REEXEC:-}" ]]; then
  REEXEC=0
  for _arg in "$@"; do
    case "$_arg" in
      --repo|--all) REEXEC=1 ;;
      --keep-repo) REEXEC=0 ;;
    esac
  done
  if (( REEXEC )); then
    _copy="$(mktemp "${TMPDIR:-/tmp}/nyra-uninstall-XXXXXX.sh")"
    cp "$SCRIPT_PATH" "$_copy"
    export NYRA_UNINSTALL_REEXEC=1 NYRA_UNINSTALL_COPY="$_copy" NYRA_UNINSTALL_REPO="$REPO_DEFAULT"
    exec bash "$_copy" "$@"
  fi
fi
trap '[[ -n "${NYRA_UNINSTALL_COPY:-}" ]] && rm -f "$NYRA_UNINSTALL_COPY"' EXIT

ASSUME_YES=0
DRY_RUN=1
DO_BUILD=0 DO_ENVS=0 DO_NPM=0 DO_UV_CACHE=0 DO_APT=0 DO_REPO=0 KEEP_REPO=0
APT_GENERIC=0
REPO="$REPO_DEFAULT"
LOG="${TMPDIR:-/tmp}/nyra-uninstall-$(date +%Y%m%d-%H%M%S).log"

usage() { sed -n '2,30p' "$SCRIPT_PATH" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while (($#)); do
  case "$1" in
    --yes|-y) ASSUME_YES=1; DRY_RUN=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --build) DO_BUILD=1; shift ;;
    --envs) DO_ENVS=1; shift ;;
    --npm) DO_NPM=1; shift ;;
    --uv-cache) DO_UV_CACHE=1; shift ;;
    --apt) DO_APT=1; shift ;;
    --apt-generic) DO_APT=1; APT_GENERIC=1; shift ;;
    --repo) DO_REPO=1; shift ;;
    --keep-repo) KEEP_REPO=1; shift ;;
    --all) DO_BUILD=1; DO_ENVS=1; DO_NPM=1; DO_UV_CACHE=1; DO_REPO=1; shift ;;
    --path) REPO="${2:?--path needs a directory}"; shift 2 ;;
    --force-path) FORCE_PATH=1; shift ;;
    --log) LOG="${2:?--log needs a file}"; shift 2 ;;
    -h|--help) usage 0 ;;
    *) echo "uninstall: unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
done

(( KEEP_REPO )) && DO_REPO=0

# The checkout must look like a Nyra checkout, and never be a dangerous root.
RESOLVED_REPO="$(readlink -f "$REPO" 2>/dev/null || printf '%s' "$REPO")"
case "$RESOLVED_REPO" in
  "/"|"$HOME"|"$(dirname "$HOME")"|"")
    echo "uninstall: refusing --path '$REPO' (not a project checkout)" >&2; exit 2 ;;
esac
if [[ ! -f "$RESOLVED_REPO/.gitmodules" ]]; then
  echo "uninstall: refusing '$RESOLVED_REPO': no .gitmodules (not a Nyra checkout). Use --force-path to override." >&2
  [[ "${FORCE_PATH:-0}" == 1 ]] || exit 2
fi
if (( !DO_BUILD && !DO_ENVS && !DO_NPM && !DO_UV_CACHE && !DO_APT && !DO_REPO )); then
  DO_BUILD=1; DO_ENVS=1; DO_NPM=1; DO_REPO=1   # sensible default plan (still dry run)
fi

C_RESET=""; C_BOLD=""; C_RED=""; C_GREEN=""; C_YELLOW=""; C_DIM=""
if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_RED=$'\033[31m'
  C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_DIM=$'\033[2m'
fi

say()   { printf '%s\n' "$*"; }
head_() { printf '\n%s== %s%s\n' "$C_BOLD" "$*" "$C_RESET"; }
ok()    { printf '  %sok%s   %s\n' "$C_GREEN" "$C_RESET" "$*"; }
skip()  { printf '  %sskip%s %s\n' "$C_DIM" "$C_RESET" "$*"; }
warn()  { printf '  %swarn%s %s\n' "$C_YELLOW" "$C_RESET" "$*"; }
fail()  { printf '  %sfail%s %s\n' "$C_RED" "$C_RESET" "$*"; }

# ---------------------------------------------------------------- bookkeeping

REMOVED=0
FREED_KB=0
FAILED=0
: > "$LOG"

human() { # KB -> human
  awk -v k="${1:-0}" 'BEGIN{ split("KB MB GB TB",u," "); i=1; while (k>=1024 && i<4){k/=1024;i++} printf "%.1f %s", k, u[i] }'
}

size_kb() { # path -> KB (0 if missing)
  [[ -e "$1" ]] || { echo 0; return; }
  du -sk "$1" 2>/dev/null | awk '{print $1}'
}

# Whitelist: only these roots may be deleted. Anything else is refused.
# Both the path as given and its resolved form are checked: ~/.cache or
# ~/.venvs may be symlinks (to another disk, for example), and resolving them
# first must not turn a whitelisted path into a refused one.
ALLOW_UV_TARGET="${ALLOW_UV_TARGET:-}"
allowed() {
  local p="$1"
  local abs="$p"
  [[ "$abs" == /* ]] || abs="$PWD/$abs"
  local resolved
  resolved="$(readlink -f "$p" 2>/dev/null || printf '%s' "$abs")"
  local repo_resolved
  repo_resolved="$(readlink -f "$REPO" 2>/dev/null || printf '%s' "$REPO")"
  local candidate
  for candidate in "$abs" "$resolved"; do
    [[ -n "$candidate" ]] || continue
    [[ "$candidate" == "/" ]] && continue
    [[ "$candidate" == "$HOME" ]] && continue
    case "$candidate" in
      "$repo_resolved"|"$repo_resolved"/*) return 0 ;;
      "$HOME/.venvs/nyra"|"$HOME/.venvs/nyra"/*) return 0 ;;
      "$HOME/.cache/uv"|"$HOME/.cache/uv"/*) return 0 ;;
      "${ALLOW_UV_TARGET:-/nonexistent}"|"${ALLOW_UV_TARGET:-/nonexistent}"/*) return 0 ;;
      "${NPM_GLOBAL_PREFIX:-/nonexistent}"/*) return 0 ;;
    esac
  done
  printf '  %sdetail%s %s (resolved: %s)\n' "$C_DIM" "$C_RESET" "$abs" "$resolved" >&2
  return 1
}

remove() { # remove <path> [label]
  local path="$1" label="${2:-$1}"
  if [[ ! -e "$path" ]]; then skip "$label (not present)"; return 0; fi
  if ! allowed "$path"; then
    fail "$label — refused: outside the allowed paths"
    printf 'refused %s\n' "$path" >> "$LOG"
    FAILED=$((FAILED + 1))
    return 1
  fi
  local kb; kb="$(size_kb "$path")"
  if (( DRY_RUN )); then
    printf '  %swould remove%s %s  (%s)\n' "$C_YELLOW" "$C_RESET" "$label" "$(human "$kb")"
    return 0
  fi
  if rm -rf -- "$path" 2>>"$LOG"; then
    ok "$label removed ($(human "$kb"))"
    printf 'removed %s (%s KB)\n' "$path" "$kb" >> "$LOG"
    REMOVED=$((REMOVED + 1)); FREED_KB=$((FREED_KB + kb))
  else
    fail "$label — rm failed (see $LOG)"
    FAILED=$((FAILED + 1))
  fi
}

# ---------------------------------------------------------------- 1. build outputs

remove_build() {
  head_ "Build outputs and generated state"
  local deploy="$REPO/00_body_base/roboparty_deploy"
  for d in build install log; do remove "$deploy/$d" "00_body_base/roboparty_deploy/$d"; done
  remove "$REPO/06_assets/build" "06_assets/build (manifests, logs)"
  remove "$REPO/06_assets/evolayer" "06_assets/evolayer (EvoLayer state, audit chain, snapshots)"
  remove "$REPO/06_assets/compat_report.json" "06_assets/compat_report.json"
  remove "$REPO/.nyra-build.json" ".nyra-build.json (remembered interpreter)"
  remove "$REPO/.pytest_cache" ".pytest_cache"
  remove "$REPO/06_evolayer/.pytest_cache" "06_evolayer/.pytest_cache"
  remove "$REPO/tools/build/.pytest_cache" "tools/build/.pytest_cache"
  local n=0
  while IFS= read -r -d '' pyc; do
    if (( DRY_RUN )); then n=$((n + 1)); else rm -rf -- "$pyc" 2>>"$LOG" && n=$((n + 1)); fi
  done < <(find "$REPO" -type d -name __pycache__ -not -path '*/.git/*' -print0 2>/dev/null)
  if (( n )); then
    if (( DRY_RUN )); then printf '  %swould remove%s %d __pycache__ directories\n' "$C_YELLOW" "$C_RESET" "$n"
    else ok "$n __pycache__ directories removed"; fi
  fi
}

# ---------------------------------------------------------------- 2. python envs

remove_envs() {
  head_ "Python environments created by this project"
  local root="$REPO"
  remove "$root/04_training/roboparty_train/.venv" "roboparty_train/.venv (Isaac Lab, torch)"
  remove "$root/04_training/roboparty_train/.venv-physx" "roboparty_train/.venv-physx"
  remove "$root/04_training/IsaacLab-3.0.0" "04_training/IsaacLab-3.0.0 (toolchain clone)"
  local module
  for module in human-humanoid-tools any4hdmi PartyMimic UFO; do
    remove "$root/02_motion_infra/$module/.venv" "02_motion_infra/$module/.venv"
  done
  remove "$root/02_motion_infra/MimicLite/active-adaptation/venv/mjlab/.venv" "MimicLite venv/mjlab/.venv"
  remove "$root/02_motion_infra/MimicLite/active-adaptation/venv/isaaclab/.venv" "MimicLite venv/isaaclab/.venv"
  remove "$root/03_advanced/PredActor/.venv" "PredActor/.venv"
  remove "$root/03_advanced/P-BFM/.venv" "P-BFM/.venv"
  remove "$root/03_advanced/I-BFM-Code/ibfm/.venv" "I-BFM/ibfm/.venv"
  remove "$root/03_advanced/SADL/.venv" "SADL/.venv"
  remove "$HOME/.venvs/nyra" "~/.venvs/nyra (EvoLayer: mujoco, onnxruntime, pytest)"
  warn "MimicLite keeps tracked manifests under active-adaptation/venv — only the inner .venv dirs are removed"
}

# ---------------------------------------------------------------- 3. npm

remove_npm() {
  head_ "npm artefacts"
  remove "$REPO/node_modules" "node_modules (local)"
  remove "$REPO/package-lock.json" "package-lock.json"
  if command -v npm >/dev/null 2>&1; then
    local gprefix groot
    gprefix="$(npm config get prefix 2>/dev/null || true)"
    groot="$(npm root -g 2>/dev/null || true)"
    NPM_GLOBAL_PREFIX="$gprefix"
    if [[ -n "$groot" && -d "$groot/nyra-workspace" ]]; then
      remove "$groot/nyra-workspace" "global npm package nyra-workspace"
    else
      skip "global nyra-workspace package (not installed)"
    fi
    if [[ -n "$gprefix" && -e "$gprefix/bin/nyra-build" ]]; then
      remove "$gprefix/bin/nyra-build" "global bin/nyra-build"
    fi
  else
    skip "npm not on PATH"
  fi
}

# ---------------------------------------------------------------- 4. uv cache

remove_uv_cache() {
  head_ "uv cache"
  local cache="${UV_CACHE_DIR:-$HOME/.cache/uv}"
  if [[ ! -e "$cache" ]]; then skip "$cache (not present)"; return; fi
  warn "this cache is shared with every other uv project on this machine"
  if [[ -L "$cache" ]]; then
    # rm -rf on a symlink removes only the link; the data (and the space) stays.
    ALLOW_UV_TARGET="$(readlink -f "$cache")"
    warn "$cache is a symlink to $ALLOW_UV_TARGET — removing the target too"
    remove "$ALLOW_UV_TARGET" "$ALLOW_UV_TARGET (uv cache target)"
    remove "$cache" "$cache (symlink)"
  else
    remove "$cache" "$cache"
  fi
}

# ---------------------------------------------------------------- 5. apt

APT_ROS=(ros-jazzy-ros-base ros-jazzy-joy python3-colcon-common-extensions)
APT_GENERIC_PKGS=(ccache libfmt-dev libspdlog-dev libeigen3-dev screen zlib1g-dev git-lfs)

remove_apt() {
  head_ "System packages (apt)"
  if ! command -v dpkg >/dev/null 2>&1; then skip "not a dpkg system"; return; fi
  local pkgs=("${APT_ROS[@]}")
  (( APT_GENERIC )) && pkgs+=("${APT_GENERIC_PKGS[@]}")
  local installed=()
  local p
  for p in "${pkgs[@]}"; do
    if dpkg -s "$p" >/dev/null 2>&1; then installed+=("$p"); fi
  done
  if ((${#installed[@]} == 0)); then skip "no project packages installed"; else
    printf '  %swould remove%s %s\n' "$C_YELLOW" "$C_RESET" "${installed[*]}"
  fi
  local files=(/etc/apt/sources.list.d/ros2.list /usr/share/keyrings/ros-archive-keyring.gpg)
  local f
  for f in "${files[@]}"; do
    [[ -e "$f" ]] && printf '  %swould remove%s %s (ROS apt source)\n' "$C_YELLOW" "$C_RESET" "$f"
  done
  if (( DRY_RUN )); then return; fi
  if ((${#installed[@]})); then
    if (( EUID == 0 )); then
      apt-get remove -y "${installed[@]}" >>"$LOG" 2>&1 && ok "apt packages removed" || { fail "apt remove failed (see $LOG)"; FAILED=$((FAILED+1)); }
    elif command -v sudo >/dev/null 2>&1; then
      sudo apt-get remove -y "${installed[@]}" >>"$LOG" 2>&1 && ok "apt packages removed" || { fail "apt remove failed (see $LOG)"; FAILED=$((FAILED+1)); }
    else
      fail "no root/sudo — run manually: sudo apt remove ${installed[*]}"
      FAILED=$((FAILED+1))
    fi
  fi
  for f in "${files[@]}"; do
    if [[ -e "$f" ]]; then
      if (( EUID == 0 )); then rm -f -- "$f"; else sudo rm -f -- "$f"; fi
      ok "removed $f"
    fi
  done
  warn "not running 'apt autoremove' — review leftover dependencies yourself if you want them gone"
}

# ---------------------------------------------------------------- 6. the checkout

remove_repo() {
  head_ "The checkout itself"
  local kb; kb="$(size_kb "$REPO")"
  if (( !DO_REPO )); then skip "kept by request (--keep-repo)"; return; fi
  if [[ "$(readlink -f "$REPO")" == "$(readlink -f "$HOME")" ]]; then
    fail "refusing to delete your home directory"; return
  fi
  if (( DRY_RUN )); then
    printf '  %swould remove%s %s  (%s)\n' "$C_RED" "$C_RESET" "$REPO" "$(human "$kb")"
    return
  fi
  ok "removing $REPO ($(human "$kb"))"
  printf 'removed repo %s (%s KB)\n' "$REPO" "$kb" >> "$LOG"
  rm -rf -- "$REPO" 2>>"$LOG" && { FREED_KB=$((FREED_KB + kb)); REMOVED=$((REMOVED + 1)); ok "checkout removed"; } \
    || { fail "could not remove $REPO"; FAILED=$((FAILED + 1)); }
}

# ---------------------------------------------------------------- main

say "${C_BOLD}Nyra uninstaller${C_RESET}"
say "repo: $REPO"
say "log:  $LOG"
if (( DRY_RUN )); then
  say "${C_YELLOW}DRY RUN — nothing will be deleted. Re-run with --yes to execute.${C_RESET}"
else
  say "${C_RED}EXECUTING — this deletes files. Ctrl-C in the next 5 s to abort.${C_RESET}"
  sleep 5
fi

(( DO_BUILD ))    && remove_build
(( DO_ENVS ))     && remove_envs
(( DO_NPM ))      && remove_npm
(( DO_UV_CACHE )) && remove_uv_cache
(( DO_APT ))      && remove_apt
(( DO_REPO ))     && remove_repo

head_ "Summary"
if (( DRY_RUN )); then
  say "  dry run: nothing was deleted"
  say "  re-run with: $(basename "$SCRIPT_PATH") --yes$([[ $DO_APT == 1 ]] && echo ' --apt')"
else
  say "  removed: $REMOVED path(s), freed $(human "$FREED_KB")"
  (( FAILED )) && say "  ${C_RED}failed: $FAILED${C_RESET} (see $LOG)"
fi
say "  left untouched: git config, python/node/uv binaries, other projects, ~/.local/bin"
say "  log: $LOG"
say ""
say "  to reinstall later: git clone --recursive https://github.com/DanijelTech/Nyra.git && cd Nyra && npm run build"
(( FAILED )) && exit 1
exit 0
