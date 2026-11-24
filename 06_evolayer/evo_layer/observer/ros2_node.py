"""Live ROS 2 observer node.

Subscribes to the topics the deploy stack already publishes, assembles episodes
with the same rules as the offline path, and appends them to a JSONL file that
``evolayer observe`` ingests. It writes telemetry only — it never commands the
robot.

Topics (all overridable via CLI):

===================  =====================  ==================================
topic                type                   used for
===================  =====================  ==================================
``/joint_states``    sensor_msgs/JointState  effort -> torque peak, velocity peak
``/imu/data``        sensor_msgs/Imu         roll/pitch -> fall detection
``/odom``            nav_msgs/Odometry       base height -> fall detection
``/cmd_vel``         geometry_msgs/Twist     recorded command (context)
``/diagnostics``     diagnostic_msgs/...     policy version, active policy name
===================  =====================  ==================================

Run on the robot controller (ROS 2 Humble/Jazzy, Python 3.10):

    python3 -m evo_layer.observer.ros2_node --skill locomotion --duration 600
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from .ingest import EpisodeAssembler, Sample

try:  # rclpy is only present on the robot / in a sourced ROS 2 environment.
    import rclpy
    from diagnostic_msgs.msg import DiagnosticArray
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Imu, JointState

    ROS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on machines without ROS 2
    ROS_AVAILABLE = False
    Node = object  # type: ignore[assignment,misc]


def quaternion_to_roll_pitch(x: float, y: float, z: float, w: float) -> tuple[float, float]:
    """Roll/pitch (radians) from a unit quaternion in ROS order (x, y, z, w)."""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    return roll, pitch


class _Accumulator:
    """Latest value per channel, plus the peak magnitudes seen in this window."""

    def __init__(self) -> None:
        self.torque_peak = 0.0
        self.vel_peak = 0.0
        self.battery_min: float | None = None
        self.tracking_error: float | None = None
        self.roll = 0.0
        self.pitch = 0.0
        self.height: float | None = None
        self.command: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.policy_version = "unknown"

    def sample(self, t: float) -> Sample:
        return Sample(
            t=t,
            base_height=self.height,
            base_roll=self.roll,
            base_pitch=self.pitch,
            torque_nm=self.torque_peak,
            joint_vel=self.vel_peak,
            battery_v=self.battery_min,
            tracking_error=self.tracking_error,
            command=self.command,
        )


def make_node_class():  # pragma: no cover - requires ROS 2
    """Build the node class lazily so the module imports without rclpy."""

    class ObserverNode(Node):  # type: ignore[misc,valid-type]
        def __init__(self, args: argparse.Namespace) -> None:
            super().__init__("evolayer_observer")
            self.args = args
            self.acc = _Accumulator()
            self.samples: list[Sample] = []
            self.window_start = time.time()
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=20,
            )
            self.create_subscription(JointState, args.joint_states_topic, self._on_joint_state, qos)
            self.create_subscription(Imu, args.imu_topic, self._on_imu, qos)
            self.create_subscription(Odometry, args.odom_topic, self._on_odom, qos)
            self.create_subscription(Twist, args.cmd_topic, self._on_cmd, qos)
            self.create_subscription(DiagnosticArray, args.diag_topic, self._on_diagnostics, qos)
            self.timer = self.create_timer(args.flush_s, self._flush)
            self.get_logger().info(
                f"observing skill={args.skill} -> {args.out} (window {args.flush_s}s)"
            )

        # -- callbacks ---------------------------------------------------
        def _on_joint_state(self, msg: Any) -> None:
            if msg.effort:
                self.acc.torque_peak = max(self.acc.torque_peak, max(abs(e) for e in msg.effort))
            if msg.velocity:
                self.acc.vel_peak = max(self.acc.vel_peak, max(abs(v) for v in msg.velocity))

        def _on_imu(self, msg: Any) -> None:
            q = msg.orientation
            roll, pitch = quaternion_to_roll_pitch(q.x, q.y, q.z, q.w)
            self.acc.roll, self.acc.pitch = roll, pitch

        def _on_odom(self, msg: Any) -> None:
            self.acc.height = float(msg.pose.pose.position.z)

        def _on_cmd(self, msg: Any) -> None:
            self.acc.command = (float(msg.linear.x), float(msg.linear.y), float(msg.angular.z))

        def _on_diagnostics(self, msg: Any) -> None:
            for status in msg.status:
                for kv in status.values:
                    key = kv.key.lower()
                    if key in ("policy", "policy_name", "policy_version", "model"):
                        self.acc.policy_version = str(kv.value)
                    elif key in ("battery_min_v", "battery_voltage"):
                        try:
                            self.acc.battery_min = float(kv.value)
                        except ValueError:
                            continue

        # -- window handling --------------------------------------------
        def _flush(self) -> None:
            now = time.time()
            if not self.samples:
                self.window_start = now
            self.samples.append(self.acc.sample(now))
            # Keep a sliding window of samples; assemble a short episode per flush.
            if now - self.window_start < self.args.flush_s:
                return
            assembler = EpisodeAssembler(
                skill=self.args.skill,
                policy_version=self.acc.policy_version,
                source=self.args.source,
                tilt_rad=self.args.tilt_rad,
                height_floor=self.args.height_floor,
                min_duration_s=self.args.min_duration_s,
                idle_gap_s=self.args.flush_s * 4,
            )
            episodes = assembler.assemble(self.samples, require_success=True)
            path = Path(self.args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                for episode in episodes:
                    record = {
                        "record": "episode",
                        "schema_version": 1,
                        "episode_id": episode.id,
                        "skill": episode.skill,
                        "policy_version": episode.policy_version,
                        "source": episode.source,
                        "started_at": episode.started_at,
                        "duration_s": episode.duration_s,
                        "success": episode.success,
                        "falls": episode.falls,
                        "torque_peak_nm": episode.torque_peak_nm,
                        "joint_vel_peak_rps": episode.joint_vel_peak_rps,
                        "collisions": episode.collisions,
                        "battery_min_v": episode.battery_min_v,
                        "tracking_error_rad": episode.tracking_error_rad,
                        "notes": episode.notes,
                    }
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
            self.get_logger().info(f"wrote {len(episodes)} episode(s) to {path}")
            self.samples = []
            self.window_start = now

    return ObserverNode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skill", required=True, help="skill id these episodes belong to")
    parser.add_argument("--out", required=True, help="JSONL file to append episodes to")
    parser.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (0 = forever)")
    parser.add_argument("--flush-s", type=float, default=10.0, help="episode window length")
    parser.add_argument("--min-duration-s", type=float, default=2.0)
    parser.add_argument("--tilt-rad", type=float, default=0.7, help="fall threshold on |roll|/|pitch|")
    parser.add_argument("--height-floor", type=float, default=0.25, help="fall threshold on base height")
    parser.add_argument("--source", default="real", choices=["real", "sim", "replay"])
    parser.add_argument("--joint-states-topic", default="/joint_states")
    parser.add_argument("--imu-topic", default="/imu/data")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--cmd-topic", default="/cmd_vel")
    parser.add_argument("--diag-topic", default="/diagnostics")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not ROS_AVAILABLE:
        print(
            "rclpy is not importable. Source the ROS 2 environment first:\n"
            "  source /opt/ros/humble/setup.bash",
            file=sys.stderr,
        )
        return 2
    rclpy.init()  # pragma: no cover - requires ROS 2
    node = make_node_class()(args)  # pragma: no cover
    started = time.time()
    try:  # pragma: no cover
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=1.0)
            if args.duration and time.time() - started >= args.duration:
                node._flush()
                break
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:  # pragma: no cover
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
