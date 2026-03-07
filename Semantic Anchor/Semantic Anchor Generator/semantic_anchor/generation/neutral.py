import json

from ..llm import call_llm, _role_context

BOLD = "\033[1m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def generate_neutral_anchors(proposition, config, role="user"):
    """
    Generate NEUTRAL anchors — everyday messages completely outside the
    proposition's domain. Used as a baseline: if a message is closer to
    neutral than to positive anchors, it's clearly off-topic.

    Example for "hack into an account":
      Neutral: "What's a good recipe for pasta?", "Help me plan a road trip"
    """
    rc = _role_context(role)
    print("\n  {}Neutral anchors:{} Generating off-topic baseline examples...".format(
        BOLD, RESET))

    neutral_prompt = (
        'PROPOSITION (a topic domain): "{}"\n\n'
        'Generate 30 short, everyday {} that are COMPLETELY UNRELATED to this '
        'proposition and its domain. These should be normal things {}:\n\n'
        '- Cooking, recipes, food\n'
        '- Travel, directions, weather\n'
        '- Creative writing, poetry, stories\n'
        '- Math, science, history\n'
        '- Shopping, product recommendations\n'
        '- Health, fitness, general advice\n'
        '- Entertainment, movies, music, games\n'
        '- Work productivity, emails, scheduling\n'
        '- Programming (unrelated to proposition domain)\n'
        '- Random curiosity questions\n\n'
        'RULES:\n'
        '- NONE of these should use vocabulary from the proposition domain\n'
        '- Mix very short (3-5 words) and medium length (8-15 words)\n'
        '- Keep them natural\n\n'
        'Output ONLY valid JSON:\n'
        '{{"neutral": ["example1", "example2", ...]}}'
    ).format(proposition, rc["example_noun_short"], rc["author_desc"])

    neutral_system = (
        "Generate a list of everyday, off-topic questions and requests. "
        "These must be completely unrelated to the given proposition. "
        "Output ONLY valid JSON."
    )

    result = call_llm(config, neutral_system, neutral_prompt)

    try:
        if isinstance(result, dict):
            parsed = result
        else:
            parsed = json.loads(str(result).strip())
        neutral_list = parsed.get("neutral", [])
        neutral_list = [str(e).strip() for e in neutral_list if str(e).strip()]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        print("    {}WARNING: Failed to parse neutral anchors: {}{}".format(YELLOW, e, RESET))
        neutral_list = []

    print("    Generated {} neutral anchors".format(len(neutral_list)))
    return neutral_list
