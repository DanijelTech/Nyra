# nyra-build — one command that builds the workspace

```bash
npm install          # no dependencies: nothing is downloaded
npx nyra-build plan  # what would run here, and why something cannot
npx nyra-build build # build everything this machine supports
npx nyra-build status
```

`npm run build` is the same thing. The package is `private: true`: it is meant to be
run inside a checkout, not published to a registry.

## What "build" means

Eight stages, wired as a dependency graph. Each stage either **runs**, is
**skipped with a reason**, or **fails**. Nothing is silently green.

| # | Stage | What it does | Needs | Notes |
|---|---|---|---|---|
| 1 | `submodules` | `git submodule sync/update --init --recursive` (40 modules, nested) | git | also maps MimicLite's SSH URLs to HTTPS |
| 2 | `assets` | parses `rpo.urdf` + `mjcf/rpo.xml`, checks joint sets match and every referenced mesh exists | — | pure Python, no toolchain, always available |
| 3 | `deploy` | `colcon build --symlink-install --base-paths src --packages-up-to roboparty_inference` | ROS 2 + colcon | the C++ inference node |
| 4 | `motion` | `05_integration/scripts/setup_all.sh --motion` | uv | **opt-in**, ≈15 GB |
| 5 | `training` | `05_integration/scripts/setup_all.sh --train` | uv + `04_training/IsaacLab-3.0.0` | **opt-in**, ≈10 GB |
| 6 | `contract` | `05_integration/scripts/check_compat.py` (URDF ↔ MJCF ↔ UFO ↔ deploy ↔ train) | pyyaml | the compatibility gate from the workspace README |
| 7 | `evolayer` | `pytest 06_evolayer/tests` (145 tests) | pytest | `--skip-tests` to omit |
| 8 | `manifest` | writes `06_assets/build/manifest.json` + history | — | always last, records itself |

Stages 4 and 5 install Python environments. They are **opt-in** because they are
the only stages that download gigabytes:

```bash
npx nyra-build build --with-envs              # install motion + training envs
npx nyra-build build --only motion --with-envs # just the motion tier
npx nyra-build build --with-envs --min-free-gb 20
```

Before an environment stage starts, the builder checks free disk space and
**refuses** (exit 2) if it cannot fit — with the numbers and the options in the
message. A build that fills the disk is an incident, not a build.

## Commands

| Command | Purpose |
|---|---|
| `plan` | print the graph, dependency order and capability availability (`--json`) |
| `build` | run every supported stage (`--jobs`, `--dry-run`, `--force`, `--strict`, `--only`, `--skip`) |
| `verify` | fast pre-flight: `assets` + `contract` only |
| `test` | both pytest suites (`06_evolayer/tests`, `tools/build/tests`) |
| `doctor` | toolchain, capabilities, module state, EvoLayer self-check (`--json`) |
| `status` | the last manifest (`--json`) |
| `clean` | remove build outputs; `--venvs` also removes the multi-GB environments |

Useful flags:

* `--dry-run` — print every command, execute none, write no manifest;
* `--force` — ignore cached stage results;
* `--strict` — exit 1 when a **required** stage was skipped (CI);
* `--json` — machine-readable output for the same run;
* `--jobs N` — run independent stages in parallel (default 1: these stages are I/O
  heavy and mostly serial anyway).

Exit codes: `0` ok, `1` a stage failed (or `--strict` with a required skip),
`2` refused before starting (disk, bad flags).

## Caching

Each stage records a fingerprint: the HEAD of the repos it depends on, the skill
configs, and the builder version. A second `build` reports `cached` for unchanged
stages instead of re-running them; `--force` overrides. The fingerprint is stored
in the manifest, so the decision is auditable rather than magic.

## The manifest

`06_assets/build/manifest.json` is the record of a build:

```json
{
  "schema": 1, "builder": "1.0.0", "status": "degraded",
  "git": {"branch": "main", "head": "…", "dirty": false},
  "toolchain": {"python": "3.13.15", "node": "v24.22.2", "npm": "12.2.2"},
  "capabilities": {"colcon": {"available": false, "detail": "colcon not found on PATH"}},
  "stages": [{"name": "deploy", "status": "skipped", "reason": "missing capability — colcon: …"}],
  "summary": {"ok": 5, "skipped": 3, "failed": 0, "cached": 0}
}
```

`06_assets/build/manifest-history.jsonl` appends one line per run (status, summary,
git), so a CI job can diff builds. Per-run logs are `build-YYYYmmdd-HHMMSS.log`.

## Why Python behind an npm launcher

The workspace is ROS 2 + C++ + Python + uv. Node cannot drive `colcon` or `uv`
better than a process launcher can, and re-implementing the pipeline in JavaScript
would create two definitions of "built" that drift. So:

```
package.json  →  tools/build/cli.js  →  tools/build/nyra_build.py  →  the stack
   (bin)          (spawn, 0 deps)        (the only pipeline definition)
```

The launcher has **no dependencies**, forwards argv/stdio and returns the child's
exit code (and re-raises signals, so Ctrl-C behaves). Set `NYRA_PYTHON` to choose
the interpreter, `NYRA_ROOT` to pin the workspace.

## Requirements

* Python 3.10+ (stdlib only for the builder itself).
* git, and a configured git identity if you will run EvoLayer deploys.
* ROS 2 + colcon for the `deploy` stage; uv for the environment stages;
  pytest + pyyaml for `contract`/`evolayer`.
* Node 18+ only for the npm entry point — `python3 tools/build/nyra_build.py …`
  works without Node at all.

## Tests

```bash
python3 -m pytest tools/build/tests -q     # 28 tests: graph, stages, cache, disk guard, clean, npm launcher
npx nyra-build test                        # both suites
```

The tests build a throw-away workspace (real git, real files) and drive the
builder through its public API; the Node launcher is exercised as a subprocess.
