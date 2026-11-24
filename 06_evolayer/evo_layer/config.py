"""Layered configuration for EvoLayer.

Precedence: built-in defaults → ``config/evolayer.yaml`` → ``config/skills/*.yaml``
→ environment (``EVOLAYER_*``) → explicit overrides (used by tests and the CLI).

Every value that gates a deployment (limits, allowlists, Hermes secrets) is
validated here, once, so no module ever has to re-check a config invariant.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(RuntimeError):
    """Raised with every validation problem at once."""


# --------------------------------------------------------------------------- pieces


@dataclass
class Limits:
    """Hard safety limits. A deployment never crosses these, whatever the statistics say."""

    fall_rate_max: float = 0.02
    torque_ratio_max: float = 1.0
    joint_vel_ratio_max: float = 1.0
    collision_rate_max: float = 0.50
    battery_min_v: float = 46.0
    success_drop_max: float = 0.02
    tracking_degradation_max: float = 0.10
    min_episodes: int = 50
    stat_alpha: float = 0.05

    def merged(self, override: Mapping[str, Any] | None) -> "Limits":
        data = dataclasses.asdict(self)
        for key, value in (override or {}).items():
            if key not in data:
                raise ConfigError(f"unknown limit {key!r}")
            data[key] = value
        return Limits(**data)

    def validate(self) -> list[str]:
        issues: list[str] = []
        for name in ("fall_rate_max", "success_drop_max", "tracking_degradation_max", "stat_alpha"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                issues.append(f"limits.{name} must be within [0, 1], got {value}")
        if self.torque_ratio_max <= 0 or self.joint_vel_ratio_max <= 0:
            issues.append("limits torque/velocity ratios must be > 0")
        if self.min_episodes < 1:
            issues.append("limits.min_episodes must be >= 1")
        return issues


@dataclass
class RepoSpec:
    """A git repository EvoLayer is allowed to change, plus its file allowlist."""

    key: str
    path: Path
    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)
    writable: bool = True

    def allows(self, relpath: str) -> bool:
        rel = relpath.replace(os.sep, "/").lstrip("./")
        for pattern in self.deny:
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(rel, pattern.rstrip("/") + "/*"):
                return False
        for pattern in self.allow:
            if fnmatch.fnmatch(rel, pattern):
                return True
            if pattern.endswith("/**") and (
                rel == pattern[:-3] or rel.startswith(pattern[:-2])
            ):
                return True
        return False

    def validate(self) -> list[str]:
        issues: list[str] = []
        if not self.path.exists():
            issues.append(f"repo {self.key!r}: path does not exist: {self.path}")
        elif not (self.path / ".git").exists():
            issues.append(f"repo {self.key!r}: not a git checkout: {self.path}")
        if not self.allow:
            issues.append(f"repo {self.key!r}: empty allowlist would permit no edits")
        return issues


@dataclass
class HermesConfig:
    bind: str = "127.0.0.1"
    port: int = 8765
    token_env: str = "EVOLAYER_HERMES_TOKEN"
    secret_env: str = "EVOLAYER_HERMES_SECRET"
    token_ttl_s: int = 900
    require_auth: bool = True
    #: Require a *single-use decision token* in addition to the transport auth.
    #: Off by default: the bearer/signature already authenticates the caller.
    require_decision_token: bool = False
    max_body_bytes: int = 65536
    #: How reviews reach your Hermes: ``webhook`` (EvoLayer POSTs) or ``pull``
    #: (Hermes polls GET /api/pending). See docs/HERMES_CONTRACT.md.
    outbound: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        issues: list[str] = []
        if self.port != 0 and not 1 <= self.port <= 65535:
            issues.append(f"hermes.port out of range: {self.port} (0 = ephemeral, for tests)")
        if self.token_ttl_s < 30:
            issues.append("hermes.token_ttl_s must be >= 30")
        if self.bind not in ("127.0.0.1", "localhost", "::1") and self.require_auth is False:
            issues.append("hermes.require_auth may only be disabled on a loopback bind")
        mode = str(self.outbound.get("mode", "pull"))
        if mode not in ("webhook", "pull"):
            issues.append(f"hermes.outbound.mode must be webhook|pull, got {mode!r}")
        if mode == "webhook":
            url = str(self.outbound.get("url", ""))
            if not url.startswith(("http://", "https://")):
                issues.append("hermes.outbound.url must be an http(s) URL when mode=webhook")
        return issues


@dataclass
class SimulatorConfig:
    default_backend: str = "mujoco"
    max_episodes: int = 2000
    timeout_s: int = 7200
    workdir: Path | None = None
    backends: dict[str, dict[str, Any]] = field(default_factory=dict)

    def backend(self, name: str | None = None) -> dict[str, Any]:
        key = name or self.default_backend
        if key not in self.backends:
            raise ConfigError(f"unknown simulator backend {key!r} (known: {sorted(self.backends)})")
        return dict(self.backends[key])

    def validate(self) -> list[str]:
        issues: list[str] = []
        if self.default_backend not in self.backends:
            issues.append(f"simulator.default_backend {self.default_backend!r} not configured")
        if self.max_episodes < 1:
            issues.append("simulator.max_episodes must be >= 1")
        if self.timeout_s < 30:
            issues.append("simulator.timeout_s must be >= 30")
        return issues


@dataclass
class LLMConfig:
    enabled: bool = False
    base_url: str = ""
    model: str = ""
    api_key_env: str = "EVOLAYER_LLM_API_KEY"
    timeout_s: int = 60
    max_calls_per_run: int = 4


@dataclass
class SkillSpec:
    """An evolvable capability with its own eval suite, backend and limits."""

    id: str
    title: str
    backend: str
    repo: str
    deploy_robot: str = "rpo"
    deploy_policy: str = ""
    train_task: str = ""
    eval_suite: dict[str, Any] = field(default_factory=dict)
    limits: Limits = field(default_factory=Limits)
    metrics: list[str] = field(default_factory=list)

    def validate(self, cfg: "EvoConfig") -> list[str]:
        issues: list[str] = []
        if not self.id:
            issues.append("skill.id is required")
        if self.backend not in cfg.simulator.backends:
            issues.append(f"skill {self.id!r}: unknown backend {self.backend!r}")
        if self.repo not in cfg.repos:
            issues.append(f"skill {self.id!r}: unknown repo {self.repo!r}")
        issues.extend(f"skill {self.id!r}: {issue}" for issue in self.limits.validate())
        return issues


# --------------------------------------------------------------------------- root


def find_root(start: Path | None = None) -> Path:
    """Locate the Nyra workspace root (the directory holding .gitmodules)."""
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / ".gitmodules").is_file():
            return candidate
    # Standalone install: two levels above this package.
    return Path(__file__).resolve().parents[2]


DEFAULT_REPOS: dict[str, dict[str, Any]] = {
    "training": {
        "path": "04_training/roboparty_train",
        "allow": [
            "robolab/robolab/tasks/**",
            "robolab/robolab/assets/**",
            "robolab/robolab/utils/**",
            "robolab/data/motions/**",
        ],
        "deny": ["robolab/robolab/tasks/**/__pycache__/**", "**/*.pyc"],
    },
    "deploy": {
        "path": "00_body_base/roboparty_deploy",
        "allow": [
            "src/inference/robots/rpo/configs/**",
            "src/inference/robots/rpo/models/**",
            "src/inference/robots/rpo/latents/**",
        ],
        "deny": ["**/*.so", "**/*.pyc"],
    },
    "evolayer": {
        "path": ".",
        "allow": ["06_evolayer/config/**", "06_evolayer/examples/**"],
        "deny": ["06_evolayer/evo_layer/**", "**/*.pyc"],
    },
}

DEFAULT_BACKENDS: dict[str, dict[str, Any]] = {
    "mujoco": {
        "kind": "mujoco",
        "cwd": "06_evolayer",
        # empty = the interpreter running the builder (where mujoco/onnxruntime live)
        "python": "",
        "mjcf": "00_body_base/rpo_description/mjcf/rpo.xml",
        "urdf": "00_body_base/rpo_description/urdf/rpo.urdf",
        "duration_s": 8.0,
        "policy_yaml": "00_body_base/roboparty_deploy/src/inference/robots/rpo/configs/default.yaml",
    },
    "fixture": {
        "kind": "fixture",
        "cwd": "06_evolayer",
        "dataset": "06_evolayer/examples/telemetry/sim_recovery.jsonl",
    },
    "isaaclab": {
        "kind": "isaaclab",
        "cwd": "04_training/roboparty_train",
        "python": ".venv/bin/python",
        "task": "RPO-AMP-Play",
        "viz": "none",
    },
    "ufo": {
        "kind": "ufo",
        "cwd": "02_motion_infra/UFO",
        "entry": "./run_inference.sh",
        "mode": "tracking",
    },
    "partymimic": {
        "kind": "partymimic",
        "cwd": "02_motion_infra/PartyMimic",
        "entry": "scripts/eval.sh",
        "sim2real": "",
    },
}

DEFAULT_SKILLS: list[dict[str, Any]] = [
    {
        "id": "locomotion",
        "title": "Locomotion (walk / run)",
        "backend": "mujoco",
        "repo": "training",
        "deploy_policy": "amp",
        "train_task": "RPO-AMP",
        "metrics": ["success_rate", "fall_rate", "torque_peak_nm", "tracking_error_rad"],
        "eval_suite": {
            "episodes": 200,
            "seeds": [0, 1, 2],
            "commands": [[0.6, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.6]],
            "motions": [],
        },
        "limits": {"fall_rate_max": 0.02, "min_episodes": 60},
    },
    {
        "id": "recovery",
        "title": "Balance recovery / get-up",
        "backend": "mujoco",
        "repo": "training",
        "deploy_policy": "getup",
        "train_task": "RPO-Getup-Mimic",
        "metrics": ["success_rate", "fall_rate", "torque_peak_nm"],
        "eval_suite": {
            "episodes": 200,
            "seeds": [0, 1, 2],
            "init_poses": ["prone", "supine", "side_left", "side_right"],
            "motions": [],
        },
        "limits": {"fall_rate_max": 0.05, "success_drop_max": 0.01, "min_episodes": 60},
    },
    {
        "id": "parkour",
        "title": "Terrain traversal (parkour)",
        "backend": "mujoco",
        "repo": "training",
        "deploy_policy": "parkour",
        "train_task": "RPO-Parkour",
        "metrics": ["success_rate", "fall_rate", "collision_rate", "torque_peak_nm"],
        "eval_suite": {
            "episodes": 150,
            "seeds": [0, 1],
            "terrains": ["stairs", "boxes", "slope"],
        },
        "limits": {"fall_rate_max": 0.08, "collision_rate_max": 1.0, "min_episodes": 40},
    },
    {
        "id": "tracking",
        "title": "Motion tracking (BeyondMimic)",
        "backend": "isaaclab",
        "repo": "training",
        "deploy_policy": "beyondmimic",
        "train_task": "RPO-BeyondMimic",
        "metrics": ["tracking_error_rad", "fall_rate", "torque_peak_nm"],
        "eval_suite": {"episodes": 100, "seeds": [0], "motions": [0, 1, 2, 3]},
        "limits": {"tracking_degradation_max": 0.10, "min_episodes": 30},
    },
]


# --------------------------------------------------------------------------- config


@dataclass
class EvoConfig:
    root: Path
    state_dir: Path
    store_path: Path
    audit_path: Path
    snapshots_dir: Path
    artifacts_dir: Path
    sim_dir: Path
    repos: dict[str, RepoSpec]
    limits: Limits
    hermes: HermesConfig
    simulator: SimulatorConfig
    skills: dict[str, SkillSpec]
    llm: LLMConfig = field(default_factory=LLMConfig)
    source_files: list[str] = field(default_factory=list)
    allow_unsafe_paths: bool = False

    # -- lookups ---------------------------------------------------------
    def repo(self, key: str) -> RepoSpec:
        try:
            return self.repos[key]
        except KeyError as exc:
            raise ConfigError(f"unknown repo {key!r} (known: {sorted(self.repos)})") from exc

    def skill(self, skill_id: str) -> SkillSpec:
        try:
            return self.skills[skill_id]
        except KeyError as exc:
            raise ConfigError(f"unknown skill {skill_id!r} (known: {sorted(self.skills)})") from exc

    def resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else (self.root / candidate)

    def rel(self, path: str | Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(self.root))
        except (ValueError, OSError):
            return str(path)

    # -- validation ------------------------------------------------------
    def validate(self) -> None:
        issues: list[str] = []
        issues.extend(self.limits.validate())
        issues.extend(self.hermes.validate())
        issues.extend(self.simulator.validate())
        for spec in self.repos.values():
            issues.extend(spec.validate())
        for skill in self.skills.values():
            issues.extend(skill.validate(self))
        if not self.allow_unsafe_paths:
            for spec in self.repos.values():
                try:
                    spec.path.resolve().relative_to(self.root.resolve())
                except ValueError:
                    issues.append(f"repo {spec.key!r} is outside the workspace: {spec.path}")
        if issues:
            raise ConfigError("invalid EvoLayer configuration:\n  - " + "\n  - ".join(issues))

    def ensure_dirs(self) -> None:
        for path in (self.state_dir, self.snapshots_dir, self.artifacts_dir, self.sim_dir):
            path.mkdir(parents=True, exist_ok=True)

    # -- construction ----------------------------------------------------
    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        *,
        overrides: Mapping[str, Any] | None = None,
        root: str | Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> "EvoConfig":
        env = dict(os.environ if env is None else env)
        base_root = Path(root).resolve() if root else find_root()
        cfg_path = Path(path) if path else None
        if cfg_path is None and env.get("EVOLAYER_CONFIG"):
            cfg_path = Path(env["EVOLAYER_CONFIG"])
        if cfg_path is None:
            candidate = base_root / "06_evolayer/config/evolayer.yaml"
            cfg_path = candidate if candidate.is_file() else None

        raw: dict[str, Any] = {}
        sources: list[str] = []
        if cfg_path is not None:
            if not cfg_path.is_file():
                raise ConfigError(f"config file not found: {cfg_path}")
            loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise ConfigError(f"config file must contain a mapping: {cfg_path}")
            raw = loaded
            sources.append(str(cfg_path))

        raw = _deep_merge(raw, dict(overrides or {}))

        resolved_root = Path(str(raw.get("root") or base_root)).expanduser().resolve()
        if root is not None:
            resolved_root = Path(root).resolve()

        state_dir = _p(resolved_root, raw.get("state_dir", "06_assets/evolayer"))
        limits = _limits(raw.get("limits") or {})

        repos_raw = raw.get("repos") or DEFAULT_REPOS
        repos: dict[str, RepoSpec] = {}
        for key, spec in repos_raw.items():
            repos[key] = RepoSpec(
                key=key,
                path=_p(resolved_root, spec["path"]),
                allow=list(spec.get("allow") or []),
                deny=list(spec.get("deny") or []),
                writable=bool(spec.get("writable", True)),
            )

        simulator_raw = raw.get("simulator") or {}
        simulator = SimulatorConfig(
            default_backend=str(simulator_raw.get("default_backend", "mujoco")),
            max_episodes=int(simulator_raw.get("max_episodes", 2000)),
            timeout_s=int(simulator_raw.get("timeout_s", 7200)),
            workdir=_p(resolved_root, simulator_raw.get("workdir", str(state_dir / "sim"))),
            backends=dict(simulator_raw.get("backends") or DEFAULT_BACKENDS),
        )

        hermes = HermesConfig(**(raw.get("hermes") or {}))
        llm = LLMConfig(**(raw.get("llm") or {}))

        skills_raw = raw.get("skills")
        skill_specs: dict[str, SkillSpec] = {}
        if skills_raw is None:
            skill_dir = resolved_root / "06_evolayer/config/skills"
            entries: list[dict[str, Any]] = []
            if skill_dir.is_dir():
                for skill_file in sorted(skill_dir.glob("*.yaml")):
                    data = yaml.safe_load(skill_file.read_text(encoding="utf-8")) or {}
                    if isinstance(data, dict):
                        entries.append(data)
                        sources.append(str(skill_file))
            if not entries:
                entries = [dict(entry) for entry in DEFAULT_SKILLS]
        elif isinstance(skills_raw, dict):
            entries = [dict(value, id=value.get("id", key)) for key, value in skills_raw.items()]
        else:
            entries = [dict(entry) for entry in skills_raw]

        for entry in entries:
            skill_id = str(entry.get("id") or "").strip()
            spec = SkillSpec(
                id=skill_id,
                title=str(entry.get("title") or skill_id),
                backend=str(entry.get("backend") or simulator.default_backend),
                repo=str(entry.get("repo") or "training"),
                deploy_robot=str(entry.get("deploy_robot") or "rpo"),
                deploy_policy=str(entry.get("deploy_policy") or ""),
                train_task=str(entry.get("train_task") or ""),
                eval_suite=dict(entry.get("eval_suite") or {}),
                limits=limits.merged(entry.get("limits") or {}),
                metrics=list(entry.get("metrics") or []),
            )
            skill_specs[spec.id] = spec

        cfg = cls(
            root=resolved_root,
            state_dir=state_dir,
            store_path=_p(resolved_root, raw.get("store") or str(state_dir / "state.db")),
            audit_path=_p(resolved_root, raw.get("audit") or str(state_dir / "audit.jsonl")),
            snapshots_dir=_p(resolved_root, raw.get("snapshots") or str(state_dir / "snapshots")),
            artifacts_dir=_p(resolved_root, raw.get("artifacts") or str(state_dir / "artifacts")),
            sim_dir=_p(resolved_root, raw.get("sim_dir") or str(state_dir / "sim")),
            repos=repos,
            limits=limits,
            hermes=hermes,
            simulator=simulator,
            skills=skill_specs,
            llm=llm,
            source_files=sources,
            allow_unsafe_paths=bool(raw.get("allow_unsafe_paths", False)),
        )
        cfg.validate()
        return cfg

    # -- secrets ---------------------------------------------------------
    def hermes_secret(self, env: Mapping[str, str] | None = None) -> str:
        env = dict(os.environ if env is None else env)
        secret = env.get(self.hermes.secret_env, "").strip()
        if len(secret) < 16:
            raise ConfigError(
                f"{self.hermes.secret_env} must hold at least 16 characters of entropy "
                "(generate with: python3 -c 'import secrets;print(secrets.token_urlsafe(48))')"
            )
        return secret

    def hermes_token(self, env: Mapping[str, str] | None = None) -> str:
        env = dict(os.environ if env is None else env)
        token = env.get(self.hermes.token_env, "").strip()
        if self.hermes.require_auth and len(token) < 16:
            raise ConfigError(f"{self.hermes.token_env} must hold at least 16 characters")
        return token


def _p(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path)


def _limits(data: Mapping[str, Any]) -> Limits:
    known = {f.name for f in dataclasses.fields(Limits)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"unknown limit keys: {sorted(unknown)}")
    return Limits(**dict(data))


def _deep_merge(base: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)  # type: ignore[arg-type]
        else:
            out[key] = value
    return out


def which_or_none(name: str) -> str | None:
    return shutil.which(name)
