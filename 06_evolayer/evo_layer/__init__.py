"""EvoLayer (Hermes-Evo).

A sandboxed evolutionary layer that sits above the Nyra stack. It observes the
robot, finds weaknesses, proposes changes, rehearses them in simulation, gates
them on hard safety limits, and hands the final decision to Hermes. Nothing
reaches the robot without an explicit, auditable Hermes approval.

Design invariants (enforced by code, not convention):

* The robot never edits itself. Proposals are artifacts; only the deploy
  installer writes into the stack, and only after an approved decision.
* Every state transition is recorded in an append-only, hash-chained audit log.
* Every numeric gate is computed from measured data, never from a claim.
* Rollback is always possible: deploy takes a verified snapshot and a git tag
  before it touches production files.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
