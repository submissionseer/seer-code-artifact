from __future__ import annotations

SYSTEM_PROMPT = """You are a careful evaluator. Follow the XML schema exactly.

Formatting rules (HARD):
- Output XML only, no extra text.
- <specific_information_required>: number facts exactly f1, f2, f3, ... (no gaps).
- <present_information>: one mapping per line "fi->pj" or "fi->pj,pk"; use NONE only if no requirements are supported.
- Only use fact IDs listed in <specific_information_required>.
- Never output fK+1 or any ID not listed in <specific_information_required>.
- If you believe there is another requirement, add it to <specific_information_required> first.
- Only cite passage IDs that exist in the provided passages.
"""

USER_PROMPT_TEMPLATE = """Given the question and retrieved passages, list the minimal disjoint requirements.
Return XML in the exact format:
<specific_information_required>
f1) ...
f2) ...
</specific_information_required>
<analysis>...</analysis>
<present_information>
f1->p2
f2->p1,p3
</present_information>
Do NOT include a <missing_information> section; it will be inferred from requirements and present mappings.

Question:
{question}

Passages:
{passages}
"""


def format_passages(passages: list[dict]) -> str:
    lines = []
    for passage in passages:
        lines.append(f"p{passage['pid']}: {passage['text']}")
    return "\n".join(lines)
