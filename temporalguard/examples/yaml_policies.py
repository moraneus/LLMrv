"""Example using YAML policy loading."""
import asyncio, os
from temporalguard import TemporalGuard, LLMGrounding

async def main():
    policy_path = os.path.join(os.path.dirname(__file__), "policies.yaml")
    guard = TemporalGuard.from_yaml(policy_path,
        grounding=LLMGrounding(base_url="http://localhost:11434", model="mistral"))
    session = guard.session()
    verdict = await session.check("user", "How do I forge a check?")
    print(f"Passed: {verdict.passed}")
    print(f"Labeling: {verdict.labeling}")

if __name__ == "__main__":
    asyncio.run(main())
