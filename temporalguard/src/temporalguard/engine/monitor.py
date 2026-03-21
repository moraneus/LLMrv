"""Monitor orchestrator."""
from __future__ import annotations
import asyncio
import uuid
from temporalguard.builtins import BUILTIN_USER_TURN
from temporalguard.engine.grounding import GroundingMethod, GroundingResult
from temporalguard.engine.ptltl import PtLTLMonitor, parse
from temporalguard.engine.trace import ConversationTrace
from temporalguard.policy import Policy, Proposition, Verdict, ViolationInfo

class ConversationMonitor:
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
        for i, policy in enumerate(policies):
            if not policy.enabled:
                continue
            key = policy.name or f"policy_{i}"
            self._policies[key] = policy
            ast = parse(policy.formula)
            self._monitors[key] = PtLTLMonitor(ast)

    async def process_message(self, role: str, text: str) -> Verdict:
        event = self.trace.append(role, text)
        relevant_props = [p for p in self._propositions.values() if p.role == role]
        labeling: dict[str, bool] = {}
        grounding_details: list[dict] = []
        if relevant_props:
            tasks = [self._safe_ground(event, prop) for prop in relevant_props]
            results = await asyncio.gather(*tasks)
            for prop, result in zip(relevant_props, results, strict=True):
                labeling[prop.prop_id] = result.match
                grounding_details.append(result.to_dict())
        labeling[BUILTIN_USER_TURN] = role == "user"
        for prop_id in self._propositions:
            if prop_id not in labeling:
                labeling[prop_id] = False
        per_policy: dict[str, bool] = {}
        violations: list[ViolationInfo] = []
        for policy_key, monitor in self._monitors.items():
            was_already_violated = not monitor.verdict
            verdict = monitor.step(labeling)
            per_policy[policy_key] = verdict
            if not verdict:
                policy = self._policies[policy_key]
                violation_details = list(grounding_details)
                if was_already_violated:
                    violation_details = [
                        {"match": False, "confidence": 1.0, "reasoning": "This policy was violated at a previous step. H(·) violations are irrevocable.", "method": "monitor_note", "prop_id": "_violation_history"},
                        *violation_details,
                    ]
                violations.append(ViolationInfo(
                    policy_name=policy.name,
                    formula=policy.formula,
                    violated_at_index=event.index,
                    labeling=dict(labeling),
                    grounding_details=violation_details,
                ))
        overall = all(per_policy.values()) if per_policy else True
        return Verdict(
            passed=overall,
            violations=violations,
            per_policy=per_policy,
            labeling=labeling,
            grounding_details=grounding_details,
            trace_index=event.index,
        )

    async def _safe_ground(self, event, prop: Proposition) -> GroundingResult:
        try:
            return await self._grounding.evaluate(event, prop)
        except Exception:
            return GroundingResult(match=False, confidence=0.0, reasoning="Grounding error (fail-open)", method="error", prop_id=prop.prop_id)

    def reset(self) -> None:
        self.trace = ConversationTrace(session_id=self.trace.session_id)
        for monitor in self._monitors.values():
            monitor.reset()
