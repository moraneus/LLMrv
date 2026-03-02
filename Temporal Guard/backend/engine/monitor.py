"""
Monitor orchestrator.

Coordinates grounding and ptLTL evaluation for each message in a
conversation. Grounds propositions, steps the ptLTL monitors, and
returns a per-policy verdict.
"""

from __future__ import annotations

import asyncio
import uuid

from backend.engine.grounding import GroundingMethod, GroundingResult
from backend.engine.ptltl import PtLTLMonitor, parse
from backend.engine.trace import ConversationTrace
from backend.models.policy import MonitorVerdict, Policy, Proposition, ViolationInfo


class ConversationMonitor:
    """Orchestrates runtime verification for one conversation session.

    For each message:
    1. Append to trace
    2. Filter propositions by role constraint
    3. Evaluate matching propositions via grounding engine
    4. Feed boolean labeling to ptLTL monitors
    5. Return aggregated verdict
    """

    def __init__(
        self,
        policies: list[Policy],
        propositions: list[Proposition],
        grounding: GroundingMethod,
        session_id: str = "",
    ) -> None:
        self._grounding = grounding
        self._propositions = {p.prop_id: p for p in propositions}
        self._policies: dict[str, Policy] = {}
        self._monitors: dict[str, PtLTLMonitor] = {}
        self.trace = ConversationTrace(session_id=session_id or str(uuid.uuid4()))
        # Carry forward last grounded value for each proposition across steps.
        # This allows cross-role temporal properties (e.g., p_weapon=T at user step
        # carries into assistant step for H(p_weapon -> !q_comply) to detect violations).
        self._last_labeling: dict[str, bool] = {p.prop_id: False for p in propositions}

        for policy in policies:
            if not policy.enabled:
                continue
            self._policies[policy.policy_id] = policy
            ast = parse(policy.formula_str)
            self._monitors[policy.policy_id] = PtLTLMonitor(ast)

    async def process_message(self, role: str, text: str) -> MonitorVerdict:
        """Process a new message through the full RV pipeline.

        Returns:
            MonitorVerdict with overall pass/block decision,
            per-policy verdicts, and grounding details.
        """
        # 1. Append to trace
        event = self.trace.append(role, text)

        # 2. Collect propositions matching this role
        relevant_props = [p for p in self._propositions.values() if p.role == role]

        # 3. Ground each relevant proposition (in parallel)
        labeling: dict[str, bool] = {}
        grounding_details: list[dict] = []

        if relevant_props:
            grounding_tasks = []
            for prop in relevant_props:
                grounding_tasks.append(self._safe_ground(event, prop))

            results = await asyncio.gather(*grounding_tasks)
            for prop, result in zip(relevant_props, results, strict=True):
                labeling[prop.prop_id] = result.match
                grounding_details.append(result.to_dict())

        # 4. Non-matching roles retain their last grounded value.
        # This enables cross-role temporal properties like H(p_weapon -> !q_comply)
        # where p_weapon (user-role) was True at a previous step.
        # Include carried-forward propositions in grounding_details so the user
        # can see the full labeling picture (not just role-matched ones).
        for prop_id in self._propositions:
            if prop_id not in labeling:
                carried_value = self._last_labeling.get(prop_id, False)
                labeling[prop_id] = carried_value
                prop = self._propositions[prop_id]
                grounding_details.append(
                    {
                        "match": carried_value,
                        "confidence": 1.0,
                        "reasoning": (
                            f"Carried from previous {prop.role} message"
                            if carried_value
                            else f"Not evaluated ({prop.role}-role proposition, current message is {role})"
                        ),
                        "method": "carried_forward",
                        "prop_id": prop_id,
                    }
                )

        # Update last labeling for carry-forward
        self._last_labeling.update(labeling)

        # 5. Step each ptLTL monitor
        per_policy: dict[str, bool] = {}
        violations: list[ViolationInfo] = []

        for policy_id, monitor in self._monitors.items():
            was_already_violated = not monitor.verdict
            verdict = monitor.step(labeling)
            per_policy[policy_id] = verdict

            if not verdict:
                policy = self._policies[policy_id]
                # If the violation is pre-existing (H(·) irrevocability),
                # note this so the user understands the current step didn't cause it.
                violation_details = list(grounding_details)
                if was_already_violated:
                    violation_details = [
                        {
                            "match": False,
                            "confidence": 1.0,
                            "reasoning": (
                                "This policy was violated at a previous step. "
                                "H(·) violations are irrevocable — once violated, "
                                "the policy remains violated for the rest of the session."
                            ),
                            "method": "monitor_note",
                            "prop_id": "_violation_history",
                        },
                        *violation_details,
                    ]
                violations.append(
                    ViolationInfo(
                        policy_id=policy_id,
                        policy_name=policy.name,
                        formula_str=policy.formula_str,
                        violated_at_index=event.index,
                        labeling=dict(labeling),
                        grounding_details=violation_details,
                    )
                )

        # 6. Aggregate: block if ANY policy is violated
        overall = all(per_policy.values()) if per_policy else True

        return MonitorVerdict(
            passed=overall,
            per_policy=per_policy,
            labeling=labeling,
            grounding_details=grounding_details,
            trace_index=event.index,
            violations=violations,
        )

    async def _safe_ground(self, event, prop: Proposition) -> GroundingResult:
        """Ground a proposition with fail-open error handling."""
        try:
            return await self._grounding.evaluate(event, prop)
        except Exception:
            return GroundingResult(
                match=False,
                confidence=0.0,
                reasoning="Grounding error (fail-open)",
                method="error",
                prop_id=prop.prop_id,
            )

    def reset(self) -> None:
        """Reset all monitors and trace (new conversation)."""
        self.trace = ConversationTrace(session_id=self.trace.session_id)
        self._last_labeling = {p: False for p in self._propositions}
        for monitor in self._monitors.values():
            monitor.reset()
