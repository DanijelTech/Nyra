"""MuJoCo rehearsal of a candidate policy on the RPO model.

This is the GPU-free safety simulator: it loads ``rpo_description``'s MJCF, drives
the 23 torque motors with the same PD gains and action scaling that
``roboparty_deploy`` uses, and feeds the policy the same proprioceptive
observation the deploy node builds. It answers exactly one question — *does this
policy stand up, stay inside its torque and velocity limits, and follow the
commanded velocity* — which is what the safety gate needs before Hermes looks at
a change.

It is deliberately not a rendering-faithful digital twin: contacts, actuators and
sensor noise are the MJCF's, and the report says so.

Usage::

    python -m evo_layer.simulator.mujoco_runner \
        --mjcf 00_body_base/rpo_description/mjcf/rpo.xml \
        --policy-yaml 00_body_base/roboparty_deploy/src/inference/robots/rpo/configs/default.yaml \
        --policy models/policy.onnx --episodes 20 --seeds 0,1,2 \
        --command 0.6,0,0 --command 1.0,0,0 --command 0,0,0.6 \
        --duration 8 --out out/episodes.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit("numpy is required for the MuJoCo evaluator: pip install numpy") from exc

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise SystemExit("mujoco is required for the MuJoCo evaluator: pip install mujoco") from exc

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - the fixture backend covers this case
    ort = None


# --------------------------------------------------------------------------- policy spec


@dataclass
class PolicySpec:
    """Everything the evaluator needs from a deploy policy YAML."""

    obs_terms: list[tuple[str, int]]
    frame_stack: int
    stack_order: str
    decimation: int
    dt: float
    kp: list[float]
    kd: list[float]
    action_scale: list[float]
    default_angle: list[float]
    joint_limits: list[float]
    clip_cmd: list[float]
    action_alpha: float = 1.0

    @property
    def obs_dim(self) -> int:
        return sum(width for _, width in self.obs_terms) * self.frame_stack

    @classmethod
    def from_yaml(cls, path: Path) -> "PolicySpec":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        params = data["inference_node"]["ros__parameters"]
        raw_layout = params["obs_layouts"][0]
        terms: list[tuple[str, int]] = []
        for chunk in raw_layout.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            name, _, width = chunk.partition(":")
            width = width.split("@")[0] if "@" in width else width
            if not width.isdigit():
                raise ValueError(f"cannot parse observation term {chunk!r}")
            terms.append((name.strip(), int(width)))
        frame_stack = int((params.get("frame_stacks") or [1])[0])
        stack_order = str((params.get("obs_stack_orders") or ["frame_major"])[0])
        action_scale = params.get("action_scale") or [0.25]
        if len(action_scale) == 1:
            action_scale = action_scale * sum(width for name, width in terms if name == "dof_pos")
        return cls(
            obs_terms=terms,
            frame_stack=frame_stack,
            stack_order=stack_order,
            decimation=int(params.get("decimation", 5)),
            dt=float(params.get("dt", 0.004)),
            kp=list(params["kp"]),
            kd=list(params["kd"]),
            action_scale=[float(v) for v in action_scale],
            default_angle=list(params["joint_default_angle"]),
            joint_limits=list(params.get("joint_limits") or []),
            clip_cmd=list(params.get("clip_cmd") or [-1, 1, -1, 1, -1, 1]),
            action_alpha=float(params.get("act_alpha", 1.0)),
        )


# --------------------------------------------------------------------------- robot


def urdf_joint_limits(path: Path) -> dict[str, tuple[float, float]]:
    """joint -> (effort_nm, velocity_rps) from the robot description."""
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        if joint.get("type") in (None, "fixed", "floating"):
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        limits[joint.get("name", "")] = (
            float(limit.get("effort", "0") or 0),
            float(limit.get("velocity", "0") or 0),
        )
    return limits


class Robot:
    """MJCF model + the joint/actuator index maps used by the controller."""

    def __init__(self, mjcf_path: Path, spec: PolicySpec, urdf_path: Path | None = None):
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)
        self.spec = spec
        self.n = len(spec.kp)

        actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)
        ]
        joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)
        ]
        actuated = [name for name in joint_names if name in actuator_names]
        if len(actuated) != self.n:
            raise SystemExit(
                f"policy expects {self.n} joints, MJCF exposes {len(actuated)} actuated joints: {actuated}"
            )
        self.joint_names = actuated
        self.qpos_adr = [int(self.model.jnt_qposadr[self.model.joint(name).id]) for name in actuated]
        self.dof_adr = [int(self.model.jnt_dofadr[self.model.joint(name).id]) for name in actuated]
        self.act_id = [self.model.actuator(name).id for name in actuated]
        self.base_qpos = 0
        self.base_dof = 0
        free = [i for i in range(self.model.njnt) if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
        if not free:
            raise SystemExit("MJCF needs a free-floating base joint")
        self.base_qpos = int(self.model.jnt_qposadr[free[0]])
        self.base_dof = int(self.model.jnt_dofadr[free[0]])
        self.timestep = float(self.model.opt.timestep)
        # Hardware limits for the per-joint ratios the safety gate consumes.
        self.effort_limits: list[float] = []
        self.velocity_limits: list[float] = []
        if urdf_path is not None:
            hardware = urdf_joint_limits(urdf_path)
            for name in actuated:
                effort, velocity = hardware.get(name, (0.0, 0.0))
                self.effort_limits.append(effort)
                self.velocity_limits.append(velocity)
        if not self.effort_limits or not all(self.effort_limits):
            ctrl = self.model.actuator_ctrlrange[np.array(self.act_id)]
            self.effort_limits = [float(max(abs(lo), abs(hi))) for lo, hi in ctrl]
            self.velocity_limits = [0.0] * self.n

    def torque_ratio(self, torque: "np.ndarray") -> float:
        ratios = [
            abs(float(value)) / limit
            for value, limit in zip(torque, self.effort_limits)
            if limit > 0
        ]
        return max(ratios) if ratios else 0.0

    def velocity_ratio(self, velocity: "np.ndarray") -> float:
        ratios = [
            abs(float(value)) / limit
            for value, limit in zip(velocity, self.velocity_limits)
            if limit > 0
        ]
        return max(ratios) if ratios else 0.0

    # -- state -----------------------------------------------------------
    def joint_pos(self) -> "np.ndarray":
        return np.array([self.data.qpos[a] for a in self.qpos_adr], dtype=np.float64)

    def joint_vel(self) -> "np.ndarray":
        return np.array([self.data.qvel[a] for a in self.dof_adr], dtype=np.float64)

    def base_quat(self) -> "np.ndarray":
        return np.array(self.data.qpos[self.base_qpos + 3 : self.base_qpos + 7], dtype=np.float64)

    def base_height(self) -> float:
        return float(self.data.qpos[self.base_qpos + 2])

    def base_velocity(self) -> "np.ndarray":
        return np.array(self.data.qvel[self.base_dof : self.base_dof + 3], dtype=np.float64)

    def gravity_body(self) -> "np.ndarray":
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, self.base_quat())
        rot = rot.reshape(3, 3)
        return rot.T @ np.array([0.0, 0.0, -1.0])

    def ang_vel_body(self) -> "np.ndarray":
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, self.base_quat())
        rot = rot.reshape(3, 3)
        world = np.array(self.data.qvel[self.base_dof + 3 : self.base_dof + 6])
        return rot.T @ world

    def roll_pitch(self) -> tuple[float, float]:
        # MuJoCo free-joint quaternions are wxyz.
        q = self.base_quat()
        w, x, y, z = q[0], q[1], q[2], q[3]
        sinr = 2.0 * (w * x + y * z)
        cosr = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr, cosr)
        sinp = 2.0 * (w * y - z * x)
        pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
        return roll, pitch

    # -- reset -----------------------------------------------------------
    def reset(self, *, height: float = 0.75, quat_wxyz: tuple[float, float, float, float] = (1, 0, 0, 0)) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.base_qpos : self.base_qpos + 3] = (0.0, 0.0, height)
        self.data.qpos[self.base_qpos + 3 : self.base_qpos + 7] = quat_wxyz
        for adr, angle in zip(self.qpos_adr, self.spec.default_angle):
            self.data.qpos[adr] = angle
        for i in range(self.model.nu):
            self.data.ctrl[i] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # -- control ---------------------------------------------------------
    def apply_targets(self, target: "np.ndarray") -> tuple[float, float, float, float]:
        """PD control to joint targets.

        Returns ``(torque_peak_nm, velocity_peak_rps, torque_ratio, vel_ratio)``.
        """
        q = self.joint_pos()
        qd = self.joint_vel()
        kp = np.array(self.spec.kp)
        kd = np.array(self.spec.kd)
        torque = kp * (target - q) - kd * qd
        ctrl_range = self.model.actuator_ctrlrange[np.array(self.act_id)]
        for i, act in enumerate(self.act_id):
            lo, hi = ctrl_range[i]
            if hi > lo:
                torque[i] = float(np.clip(torque[i], lo, hi))
        for i, act in enumerate(self.act_id):
            self.data.ctrl[act] = torque[i]
        return (
            float(np.max(np.abs(torque))),
            float(np.max(np.abs(qd))),
            self.torque_ratio(torque),
            self.velocity_ratio(qd),
        )


# --------------------------------------------------------------------------- observation


class ObservationBuilder:
    def __init__(self, spec: PolicySpec):
        self.spec = spec
        self.history: list[dict[str, "np.ndarray"]] = []
        self.last_action = np.zeros(sum(w for n, w in spec.obs_terms if n == "dof_pos"), dtype=np.float64)

    def build(self, robot: Robot, command: "np.ndarray") -> "np.ndarray":
        dof_pos = robot.joint_pos() - np.array(self.spec.default_angle)
        frame = {
            "ang_vel": robot.ang_vel_body(),
            "gravity_b": robot.gravity_body(),
            "cmd_vel": command,
            "dof_pos": dof_pos,
            "dof_vel": robot.joint_vel(),
            "last_action": self.last_action,
        }
        self.history.append(frame)
        if len(self.history) > self.spec.frame_stack:
            self.history.pop(0)
        while len(self.history) < self.spec.frame_stack:
            self.history.insert(0, {k: v.copy() for k, v in self.history[0].items()})
        ordered = [self.history[i] for i in range(len(self.history))]
        if self.spec.stack_order != "frame_major":
            ordered = list(reversed(ordered))
        parts = []
        for frame_data in ordered:
            for name, width in self.spec.obs_terms:
                value = frame_data.get(name)
                if value is None:
                    value = np.zeros(width)
                parts.append(np.asarray(value, dtype=np.float64)[:width])
        obs = np.concatenate(parts) if parts else np.zeros(0)
        if obs.shape[0] != self.spec.obs_dim:
            raise SystemExit(f"observation width {obs.shape[0]} != policy input {self.spec.obs_dim}")
        return obs


# --------------------------------------------------------------------------- policy


class OnnxPolicy:
    def __init__(self, path: Path):
        if ort is None:  # pragma: no cover
            raise SystemExit("onnxruntime is required to load a policy: pip install onnxruntime")
        if not path.is_file():
            raise SystemExit(f"policy not found: {path}")
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.input_dim = int(shape[-1]) if shape and isinstance(shape[-1], int) else None

    def check_obs_dim(self, expected: int) -> None:
        if self.input_dim is not None and self.input_dim != expected:
            raise SystemExit(
                f"policy expects {self.input_dim} observation values, but the policy YAML builds "
                f"{expected}; the ONNX and the deploy config do not match"
            )

    def act(self, obs: "np.ndarray") -> "np.ndarray":
        result = self.session.run([self.output_name], {self.input_name: obs.reshape(1, -1).astype(np.float32)})
        return np.asarray(result[0], dtype=np.float64).reshape(-1)


# --------------------------------------------------------------------------- episode


#: (roll, pitch, base height) at reset, in radians / metres.
INIT_POSES: dict[str, tuple[float, float, float]] = {
    "stand": (0.0, 0.0, 0.75),
    "prone": (math.pi / 2, 0.0, 0.30),
    "supine": (-math.pi / 2, 0.0, 0.30),
    "side_left": (0.0, math.pi / 2, 0.30),
    "side_right": (0.0, -math.pi / 2, 0.30),
}


def quat_from_roll_pitch(roll: float, pitch: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    return (cr * cp, sr * cp, cr * sp, -sr * sp)


@dataclass
class EpisodeOutcome:
    success: bool
    falls: int
    torque_peak: float
    velocity_peak: float
    duration_s: float
    tracking_error: float | None
    survived_s: float
    min_height: float
    max_tilt: float
    torque_ratio: float = 0.0
    vel_ratio: float = 0.0


def run_episode(
    robot: Robot,
    policy: OnnxPolicy | None,
    spec: PolicySpec,
    *,
    seed: int,
    command: "np.ndarray",
    init_pose: str,
    duration_s: float,
    tilt_rad: float,
    height_floor: float,
    render: Path | None = None,
    render_every: int = 5,
    settle_s: float = 0.2,
) -> EpisodeOutcome:
    if init_pose not in INIT_POSES:
        raise SystemExit(f"unknown init pose {init_pose!r} (known: {sorted(INIT_POSES)})")
    roll, pitch, height = INIT_POSES[init_pose]
    robot.reset(height=height, quat_wxyz=quat_from_roll_pitch(roll, pitch))
    rng = np.random.default_rng(seed)
    for adr, delta in zip(robot.qpos_adr, rng.normal(0.0, 0.01, len(robot.qpos_adr))):
        robot.data.qpos[adr] += float(delta)
    mujoco.mj_forward(robot.model, robot.data)
    builder = ObservationBuilder(spec)
    action = np.zeros(len(spec.kp))
    torque_peak = vel_peak = 0.0
    torque_ratio = vel_ratio = 0.0
    min_height = height
    max_tilt = 0.0
    tracking: list[float] = []
    fall_time: float | None = None
    tilt_since: float | None = None
    policy_dt = spec.dt * spec.decimation
    steps = max(1, int(duration_s / policy_dt))
    renderer = None
    frames_dir = None
    camera = None
    if render is not None:
        frames_dir = render
        frames_dir.mkdir(parents=True, exist_ok=True)
        try:
            import imageio.v3 as iio  # noqa: PLC0415 - optional dependency

            renderer = mujoco.Renderer(robot.model, height=360, width=480)
            camera = mujoco.MjvCamera()
            camera.distance = 2.5
            camera.azimuth = 90
            camera.elevation = -15
        except (ImportError, Exception) as exc:  # noqa: BLE001 - GL stack may be unavailable
            print(
                f"video capture disabled ({type(exc).__name__}: {exc}); "
                "set MUJOCO_GL=egl or osmesa and install imageio to enable it",
                file=sys.stderr,
            )
            renderer = None
            camera = None
            iio = None
    t = 0.0
    for step in range(steps):
        obs = builder.build(robot, command)
        if policy is not None:
            raw = policy.act(obs)
        else:
            raw = np.zeros(len(spec.kp))
        action = spec.action_alpha * raw + (1 - spec.action_alpha) * action
        builder.last_action = action.copy()
        target = np.array(spec.default_angle) + np.array(spec.action_scale) * action
        if spec.joint_limits:
            # joint_limits is a flat lower/upper sequence in deploy YAML order
            limits = np.array(spec.joint_limits)
            if limits.size == 2 * len(spec.kp):
                target = np.clip(target, limits[0::2], limits[1::2])
        for _ in range(spec.decimation):
            torque, vel, t_ratio, v_ratio = robot.apply_targets(target)
            mujoco.mj_step(robot.model, robot.data)
            # Ignore the first moments of contact settling: teleporting the model onto
            # the floor produces a one-step impact that says nothing about the policy.
            if t >= settle_s:
                torque_peak = max(torque_peak, torque)
                vel_peak = max(vel_peak, vel)
                torque_ratio = max(torque_ratio, t_ratio)
                vel_ratio = max(vel_ratio, v_ratio)
        t += policy_dt
        height_now = robot.base_height()
        roll_now, pitch_now = robot.roll_pitch()
        min_height = min(min_height, height_now)
        tilt = max(abs(roll_now), abs(pitch_now))
        max_tilt = max(max_tilt, tilt)
        if fall_time is None:
            if tilt > tilt_rad or height_now < height_floor:
                tilt_since = t if tilt_since is None else tilt_since
            else:
                tilt_since = None
            if tilt_since is not None and t - tilt_since >= 0.25:
                fall_time = tilt_since
                break
        if np.any(command):
            actual = robot.base_velocity()
            tracking.append(float(np.linalg.norm(actual[:2] - command[:2])))
        if renderer is not None and frames_dir is not None and iio is not None and step % max(1, render_every) == 0:
            renderer.update_scene(robot.data, camera=camera)
            iio.imwrite(frames_dir / f"{step:05d}.png", renderer.render())
    survived = t if fall_time is None else max(0.0, fall_time)
    return EpisodeOutcome(
        success=fall_time is None,
        falls=0 if fall_time is None else 1,
        torque_peak=torque_peak,
        velocity_peak=vel_peak,
        duration_s=t,
        tracking_error=(sum(tracking) / len(tracking)) if tracking else None,
        survived_s=survived,
        min_height=min_height,
        max_tilt=max_tilt,
        torque_ratio=torque_ratio,
        vel_ratio=vel_ratio,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mjcf", required=True)
    parser.add_argument("--urdf", help="URDF for per-joint effort/velocity limits (recommended)")
    parser.add_argument("--policy-yaml", required=True)
    parser.add_argument("--policy", help="ONNX policy; omit to measure the zero-action baseline")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seeds", default="0")
    parser.add_argument(
        "--command",
        dest="commands",
        action="append",
        metavar="VX,VY,WZ",
        help="commanded velocity; repeatable",
    )
    parser.add_argument("--init-poses", default="stand")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--tilt-rad", type=float, default=0.7)
    parser.add_argument("--height-floor", type=float, default=0.25)
    parser.add_argument("--settle", type=float, default=0.2, help="ignore peaks during this initial window")
    parser.add_argument("--skill", default="locomotion")
    parser.add_argument("--policy-version", default="candidate")
    parser.add_argument("--out", required=True, help="episodes.jsonl")
    parser.add_argument("--video-dir")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    spec = PolicySpec.from_yaml(Path(args.policy_yaml))
    robot = Robot(Path(args.mjcf), spec, Path(args.urdf) if args.urdf else None)
    policy = OnnxPolicy(Path(args.policy)) if args.policy else None
    if policy is not None:
        policy.check_obs_dim(spec.obs_dim)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    raw_commands = args.commands or ["0.6,0,0"]
    commands = [np.array([float(x) for x in c.split(",")]) for c in raw_commands if c.strip()]
    poses = [p.strip() for p in args.init_poses.split(",") if p.strip()]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for index in range(args.episodes):
            seed = seeds[index % len(seeds)]
            command = commands[index % len(commands)]
            pose = poses[index % len(poses)]
            render = None
            if args.video_dir and index == 0:
                render = Path(args.video_dir) / "frames"
            outcome = run_episode(
                robot,
                policy,
                spec,
                seed=seed + index,
                command=command,
                init_pose=pose,
                duration_s=args.duration,
                tilt_rad=args.tilt_rad,
                height_floor=args.height_floor,
                render=render,
                settle_s=args.settle,
            )
            record = {
                "record": "episode",
                "schema_version": 1,
                "skill": args.skill,
                "policy_version": args.policy_version,
                "source": "sim",
                "started_at": started + index,
                "duration_s": round(outcome.duration_s, 4),
                "success": outcome.success,
                "falls": outcome.falls,
                "torque_peak_nm": round(outcome.torque_peak, 3),
                "joint_vel_peak_rps": round(outcome.velocity_peak, 3),
                "collisions": 0,
                "tracking_error_rad": (
                    round(outcome.tracking_error, 5) if outcome.tracking_error is not None else None
                ),
                "torque_ratio": round(outcome.torque_ratio, 4),
                "vel_ratio": round(outcome.vel_ratio, 4),
                "notes": (
                    f"mujoco seed={seed + index} pose={pose} cmd={command.tolist()} "
                    f"survived={outcome.survived_s:.2f}s min_h={outcome.min_height:.3f} "
                    f"max_tilt={outcome.max_tilt:.3f}"
                ),
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            written += 1
    if not args.quiet:
        print(f"wrote {written} episodes to {out_path} in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
