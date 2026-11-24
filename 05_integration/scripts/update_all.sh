#!/usr/bin/env bash
# update_all.sh — keep every module at its upstream branch and report the pin drift.
#
#   ./update_all.sh --check          report local vs upstream commits, change nothing
#   ./update_all.sh                  fast-forward every submodule to its branch tip
#   ./update_all.sh --nested         also update nested submodules (deploy, train, MimicLite, firmware)
#   ./update_all.sh --remote         use `git submodule update --remote` instead of fetch+merge
#   ./update_all.sh --dry-run
#
# Nothing is committed. After a successful update, review the diff and commit the new pins:
#   git add .gitmodules 00_body_base 02_motion_infra 03_advanced 04_training
#   git commit -m "chore: bump module pins"

set -uo pipefail
# shellcheck source=lib/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

CHECK_ONLY=0
NESTED=0
USE_REMOTE=0

usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"; exit "${1:-0}"; }

while (($#)); do
  case "$1" in
    --check) CHECK_ONLY=1; shift ;;
    --nested) NESTED=1; shift ;;
    --remote) USE_REMOTE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage 0 ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
done

submodule_paths() {
  git -C "$NYRA_ROOT" config -f .gitmodules --get-regexp '^submodule\..*\.path$' | awk '{print $2}'
}

branch_of() {
  local path="$1" name branch
  name="$(git -C "$NYRA_ROOT" config -f .gitmodules --get-regexp '^submodule\..*\.path$' \
          | awk -v p="$path" '$2==p{print $1}' | sed 's/^submodule\.//; s/\.path$//')"
  branch="$(git -C "$NYRA_ROOT" config -f .gitmodules --get "submodule.$name.branch" 2>/dev/null || true)"
  printf '%s' "${branch:-HEAD}"
}

report_line() { printf '%-46s %-10s %-10s %s\n' "$1" "$2" "$3" "$4"; }

main() {
  log "workspace: $NYRA_ROOT"
  (( DRY_RUN )) && warn "dry run — nothing will be changed"

  local path local_sha remote_sha branch status dirty=() behind=()
  report_line "module" "local" "upstream" "state"

  while IFS= read -r path; do
    [[ -d "$NYRA_ROOT/$path" ]] || { report_line "$path" "-" "-" "missing"; continue; }
    branch="$(branch_of "$path")"
    local_sha="$(git -C "$NYRA_ROOT/$path" rev-parse --short=10 HEAD 2>/dev/null || echo '-')"

    if (( CHECK_ONLY )); then
      git -C "$NYRA_ROOT/$path" fetch --quiet origin "$branch" 2>/dev/null || true
      remote_sha="$(git -C "$NYRA_ROOT/$path" rev-parse --short=10 "origin/$branch" 2>/dev/null || echo '-')"
      if [[ "$local_sha" == "$remote_sha" ]]; then
        status="up to date"
      else
        status="BEHIND/AHEAD"
        [[ "$local_sha" != "$remote_sha" ]] && behind+=("$path")
      fi
    else
      if (( USE_REMOTE )); then
        local rargs=(submodule update --remote --init "$path")
        (( NESTED )) && rargs=(submodule update --remote --init --recursive "$path")
        run git -C "$NYRA_ROOT" "${rargs[@]}" || status="update failed"
      else
        run git -C "$NYRA_ROOT/$path" fetch --quiet origin "$branch" || true
        if run git -C "$NYRA_ROOT/$path" merge --ff-only "origin/$branch"; then
          status="fast-forwarded"
        else
          status="diverged (local commits?)"
        fi
      fi
      remote_sha="$(git -C "$NYRA_ROOT/$path" rev-parse --short=10 "origin/$branch" 2>/dev/null || echo '-')"
    fi

    if [[ -n "$(git -C "$NYRA_ROOT/$path" status --porcelain 2>/dev/null)" ]]; then
      dirty+=("$path")
      status="$status, dirty"
    fi
    report_line "$path" "$local_sha" "$remote_sha" "$status"
  done < <(submodule_paths)

  if (( CHECK_ONLY )); then
    step "Summary"
    if ((${#behind[@]})); then
      warn "${#behind[@]} module(s) differ from upstream:"
      printf '       - %s\n' "${behind[@]}"
      dim "run without --check to fast-forward them"
    else
      ok "all modules match their upstream branch"
    fi
  fi

  if ((${#dirty[@]})); then
    step "Dirty submodules (local changes are never touched)"
    printf '       - %s\n' "${dirty[@]}"
    dim "commit inside the module, or stash, before bumping its pin"
  fi

  if (( NESTED && ! CHECK_ONLY )); then
    step "Nested submodules"
    run git -C "$NYRA_ROOT" submodule foreach --recursive \
      'git submodule update --init --recursive || true'
  fi

  (( CHECK_ONLY )) || cat <<EOF

$(ok "pins updated locally — review and commit")
  git -C $NYRA_ROOT status --short
  git -C $NYRA_ROOT add .gitmodules 00_body_base 01_roadmap 02_motion_infra 03_advanced 04_training
  git -C $NYRA_ROOT commit -m "chore: bump module pins"
  $(rel "$NYRA_ROOT")/05_integration/scripts/check_compat.py
EOF
}

main
