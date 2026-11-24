"""Proposal templates: concrete, file-derived diffs.

Every template reads the *current* content of a file inside an allowlisted repo,
computes the new content, and emits a unified diff produced by
:mod:`difflib`. That ordering matters: the diff is guaranteed to describe the
file as it exists on disk, so validation can prove it applies, and a human
reviewing the proposal sees exactly the bytes that would change.

A template that cannot find its anchor raises :class:`TemplateUnavailable` — it
never guesses, and never falls back to a placeholder change.
"""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..config import EvoConfig, RepoSpec
from ..models import Edit, Finding


class TemplateUnavailable(RuntimeError):
    """The template cannot produce a justified change for this finding."""


@dataclass
class TemplateResult:
    kind: str
    edits: list[Edit] = field(default_factory=list)
    config_overlay: dict[str, Any] = field(default_factory=dict)
    train_command: list[str] = field(default_factory=list)
    expected_effect: str = ""
    rationale: str = ""
    sim_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Context:
    cfg: EvoConfig
    finding: Finding

    def repo(self, key: str | None = None) -> RepoSpec:
        return self.cfg.repo(key or self.cfg.skill(self.finding.skill).repo)

    def read(self, repo: RepoSpec, relpath: str) -> str:
        if not repo.allows(relpath):
            raise TemplateUnavailable(
                f"{relpath!r} is not on the {repo.key} allowlist — refusing to propose an edit"
            )
        path = repo.path / relpath
        if not path.is_file():
            raise TemplateUnavailable(f"{relpath} does not exist in {repo.path}")
        return path.read_text(encoding="utf-8")

    @staticmethod
    def diff(relpath: str, before: str, after: str) -> str:
        if before == after:
            raise TemplateUnavailable("template produced no change")
        diff = "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{relpath}",
                tofile=f"b/{relpath}",
                n=3,
            )
        )
        if not diff.strip():
            raise TemplateUnavailable("empty diff")
        return diff


# --------------------------------------------------------------------------- helpers


def _find_files(repo: RepoSpec, pattern: str, *, limit: int = 50) -> list[Path]:
    matches: list[Path] = []
    for path in sorted(repo.path.rglob(Path(pattern).name)):
        rel = str(path.relative_to(repo.path))
        if repo.allows(rel):
            matches.append(path)
        if len(matches) >= limit:
            break
    return matches


def _task_token(train_task: str) -> str:
    """``RPO-AMP`` -> ``amp``; used to prefer the matching env config file."""
    if not train_task:
        return ""
    token = train_task.split("-", 1)[-1].lower().replace("_", "")
    return token


def _format_number(literal: str, new_value: float) -> str:
    """Keep the literal's notation (scientific vs plain) without losing precision."""
    if "e" in literal.lower():
        return f"{new_value:.1e}"
    return f"{new_value:g}"


REWT_TERM = re.compile(
    r"(?P<prefix>(?P<term>[a-z0-9_]+)\s*=\s*RewTerm\((?:[^)]*?)weight\s*=\s*)(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
    re.S,
)


# --------------------------------------------------------------------------- templates


def reward_term_weight(ctx: Context) -> TemplateResult:
    """Scale one reward-term weight in the RPO training env config."""
    params = ctx.finding.params or {}
    term = str(params.get("term") or "")
    direction = str(params.get("direction") or "increase_penalty")
    if not term:
        raise TemplateUnavailable("reward_term_weight needs params.term")

    repo = ctx.repo("training")
    token = _task_token(ctx.cfg.skill(ctx.finding.skill).train_task)
    candidates: list[tuple[Path, str, re.Match[str]]] = []
    search = _find_files(repo, "rpo*_env_cfg.py") + _find_files(repo, "amp_env_cfg.py")
    search.sort(key=lambda path: (0 if token and token in path.name else 1, len(str(path))))
    for path in search:
        try:
            text = ctx.read(repo, str(path.relative_to(repo.path)))
        except TemplateUnavailable:
            continue
        for match in REWT_TERM.finditer(text):
            if match.group("term") == term:
                candidates.append((path, text, match))
    if not candidates:
        raise TemplateUnavailable(
            f"no allowed training file defines a RewTerm named {term!r} "
            "(checked robolab/robolab/tasks/**/rpo*_env_cfg.py, amp_env_cfg.py)"
        )

    path, text, match = candidates[0]
    current = float(match.group("value"))
    factor = float(params.get("factor") or (2.0 if direction == "increase_penalty" else 0.5))
    if direction == "increase_penalty":
        new_value = -abs(current) * factor if current <= 0 else current * factor
    elif direction == "decrease_penalty":
        new_value = -abs(current) / factor if current <= 0 else current / factor
    else:
        raise TemplateUnavailable(f"unknown direction {direction!r}")
    if math.isclose(new_value, current):
        raise TemplateUnavailable("computed weight equals the current weight")

    start, end = match.span("value")
    after = text[:start] + _format_number(text[start:end], new_value) + text[end:]
    relpath = str(path.relative_to(repo.path))
    return TemplateResult(
        kind="reward_term_weight",
        edits=[
            Edit(
                repo=repo.key,
                path=relpath,
                diff=ctx.diff(relpath, text, after),
                description=f"{term}: weight {current:g} -> {new_value:g}",
            )
        ],
        expected_effect=params.get("reason") or f"scales reward term {term}",
        rationale=(
            f"Reward shaping on {term} is the most direct lever for the measured failure mode; "
            f"the factor ({factor:g}x) is deliberately small enough to re-evaluate cheaply."
        ),
        sim_params={"reward_term": term, "old_weight": current, "new_weight": new_value},
    )


TRAIN_BASELINES: dict[str, Any] = {
    "--num_envs": 4096,
    "--max_iterations": 500,
    "--learning_rate": 3.0e-4,
    "--seed": 42,
}


def train_hyperparameter(ctx: Context) -> TemplateResult:
    """Increase one training budget knob, expressed as a validated CLI override."""
    params = ctx.finding.params or {}
    flag = str(params.get("flag") or "")
    scale = float(params.get("scale") or 1.0)
    if flag not in TRAIN_BASELINES:
        raise TemplateUnavailable(
            f"{flag!r} is not an allowed training flag (allowed: {sorted(TRAIN_BASELINES)})"
        )
    if flag == "--seed":
        raise TemplateUnavailable("scaling --seed is meaningless; use eval_suite_expand instead")
    if flag in ("--num_envs", "--max_iterations"):
        if not 1.0 < scale <= 4.0:
            raise TemplateUnavailable(f"{flag} may only be increased: scale {scale} outside (1, 4]")
    else:  # rates and budgets: allow a reduction, e.g. halving the learning rate
        if not 0.25 <= scale <= 4.0 or math.isclose(scale, 1.0):
            raise TemplateUnavailable(f"scale {scale} outside [0.25, 4] or is a no-op")

    base = TRAIN_BASELINES[flag]
    value: int | float = int(round(base * scale)) if isinstance(base, int) else float(base) * scale
    skill = ctx.cfg.skill(ctx.finding.skill)
    if not skill.train_task:
        raise TemplateUnavailable(f"skill {skill.id!r} has no train_task configured")

    python = ".venv/bin/python"
    command = [
        python,
        "robolab/scripts/rsl_rl/train.py",
        f"--task={skill.train_task}",
        f"{flag}={value}",
        "--viz=none",
        "--logger=tensorboard",
    ]
    return TemplateResult(
        kind="train_hyperparameter",
        train_command=command,
        config_overlay={"training": {flag.lstrip("-"): value, "task": skill.train_task}},
        expected_effect=params.get("reason") or f"raises {flag} to {value}",
        rationale=(
            f"The finding is a capability/quality gap rather than an instability; more {flag} is the "
            "cheapest validated lever. The change is a CLI override, so no source file is touched."
        ),
        sim_params={"flag": flag, "value": value},
    )


EVAL_SUITE_LIMIT = 5000


def eval_suite_expand(ctx: Context) -> TemplateResult:
    """Raise the evidence bar for this skill (more episodes, more seeds)."""
    repo = ctx.repo("evolayer")
    relpath = f"06_evolayer/config/skills/{ctx.finding.skill}.yaml"
    text = ctx.read(repo, relpath)
    match = re.search(r"(?P<prefix>^\s*episodes:\s*)(?P<value>\d+)", text, re.M)
    if not match:
        raise TemplateUnavailable(f"{relpath} has no eval_suite.episodes field")
    current = int(match.group("value"))
    target = min(EVAL_SUITE_LIMIT, current * 2)
    if target <= current:
        raise TemplateUnavailable(f"eval suite is already at the cap ({EVAL_SUITE_LIMIT} episodes)")
    start, end = match.span("value")
    after = text[:start] + str(target) + text[end:]
    return TemplateResult(
        kind="eval_suite_expand",
        edits=[
            Edit(
                repo=repo.key,
                path=relpath,
                diff=ctx.diff(relpath, text, after),
                description=f"eval episodes {current} -> {target}",
            )
        ],
        expected_effect="more episodes per rehearsal, tighter confidence interval on every gate",
        rationale=(
            "Before spending simulation on a policy change, make the measurement itself stronger. "
            "This edit only affects our own configuration, not the robot's behaviour."
        ),
        sim_params={"episodes": target},
    )


NUMERIC_YAML_KEY = re.compile(
    r"(?P<prefix>^(?P<indent>\s*)(?P<key>[a-z_]+):\s*\[?\s*)(?P<value>-?\d+(?:\.\d+)?)\s*\]?\s*$",
    re.M,
)
DEPLOY_NUMERIC_KEYS = {"action_scale": (0.01, 2.0), "act_alpha": (0.05, 1.0)}


def deploy_config_tune(ctx: Context) -> TemplateResult:
    """Tune a single numeric deploy parameter. High risk by construction."""
    params = ctx.finding.params or {}
    key = str(params.get("key") or "")
    factor = float(params.get("factor") or 1.0)
    if key not in DEPLOY_NUMERIC_KEYS:
        raise TemplateUnavailable(f"{key!r} is not a tunable deploy key (allowed: {sorted(DEPLOY_NUMERIC_KEYS)})")
    if not 0.5 <= factor <= 2.0:
        raise TemplateUnavailable("deploy tuning factor must stay within [0.5, 2.0]")

    skill = ctx.cfg.skill(ctx.finding.skill)
    policy = skill.deploy_policy
    if not policy:
        raise TemplateUnavailable(f"skill {skill.id!r} has no deploy_policy")
    repo = ctx.repo("deploy")
    relpath = f"src/inference/robots/{skill.deploy_robot}/configs/{policy}.yaml"
    text = ctx.read(repo, relpath)

    matches = [m for m in NUMERIC_YAML_KEY.finditer(text) if m.group("key") == key]
    if not matches:
        raise TemplateUnavailable(f"{relpath} has no scalar {key!r}")
    match = matches[0]
    current = float(match.group("value"))
    new_value = round(current * factor, 4)
    low, high = DEPLOY_NUMERIC_KEYS[key]
    if not low <= new_value <= high:
        raise TemplateUnavailable(f"{key} = {new_value} outside safe band [{low}, {high}]")
    if math.isclose(new_value, current):
        raise TemplateUnavailable("no change")

    start, end = match.span("value")
    after = text[:start] + f"{new_value:g}" + text[end:]
    return TemplateResult(
        kind="deploy_config_tune",
        edits=[
            Edit(
                repo=repo.key,
                path=relpath,
                diff=ctx.diff(relpath, text, after),
                description=f"{key} {current:g} -> {new_value:g}",
            )
        ],
        expected_effect=params.get("reason") or f"deploy {key} factor {factor:g}",
        rationale=(
            "This changes how the deployed controller scales actions, so it is gated as high risk, "
            "rehearsed in simulation and always reviewed by Hermes."
        ),
        sim_params={"deploy_key": key, "factor": factor},
    )


TEMPLATES: dict[str, Callable[[Context], TemplateResult]] = {
    "reward_term_weight": reward_term_weight,
    "train_hyperparameter": train_hyperparameter,
    "eval_suite_expand": eval_suite_expand,
    "deploy_config_tune": deploy_config_tune,
}


def build(template_id: str, ctx: Context) -> TemplateResult:
    try:
        template = TEMPLATES[template_id]
    except KeyError as exc:
        raise TemplateUnavailable(f"unknown template {template_id!r} (known: {sorted(TEMPLATES)})") from exc
    return template(ctx)
