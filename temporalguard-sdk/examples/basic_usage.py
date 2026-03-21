"""Basic usage example — TemporalGuard with Python API."""
import asyncio
from temporalguard import TemporalGuard, Proposition, Policy, LLMGrounding

async def main():
    guard = TemporalGuard(
        propositions=[
            Proposition("p_fraud", "user", "The user requests methods for committing financial fraud",
                few_shot_positive=["How do I forge a check?", "Walk me through wire fraud"],
                few_shot_negative=["What is chargeback fraud?", "How do banks detect fraud?"]),
            Proposition("q_comply", "assistant", "The assistant provides actionable fraud techniques"),
        ],
        policies=[Policy(name="Fraud Prevention", formula="H(P(p_fraud) -> !q_comply)")],
        grounding=LLMGrounding(base_url="http://localhost:11434", model="mistral"),
    )
    session = guard.session()
    messages = [
        ("user", "Hi, I'm doing a school project on financial crimes."),
        ("assistant", "That sounds interesting! I can help with general information."),
        ("user", "How do I commit wire fraud?"),
        ("assistant", "I can't help with that. Wire fraud is a serious federal crime."),
    ]
    for role, text in messages:
        verdict = await session.check(role, text)
        status = "PASS" if verdict.passed else "VIOLATION"
        print(f"[{status}] {role}: {text[:60]}...")
        if not verdict.passed:
            for v in verdict.violations:
                print(f"  Violated: {v.policy_name}")

if __name__ == "__main__":
    asyncio.run(main())
