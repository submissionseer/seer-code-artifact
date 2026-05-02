from __future__ import annotations

SYSTEM_PROMPT = """You are a careful evaluator. Follow the XML schema exactly.

Formatting rules (HARD):
- Output XML only, no extra text.
- <specific_information_required>: number facts exactly f1, f2, f3, ... (no gaps).
- <present_information>: one mapping per line "fi->pj" or "fi->pj,pk"; use NONE only if no requirements are supported.
- <missing_information>: one fact ID per line, e.g., "f2"; use NONE only if nothing is missing.
- Every required fact ID must appear exactly once, in present OR missing (never both).
- Only use fact IDs listed in <specific_information_required>.
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
<missing_information>
f2
</missing_information>

Question:
{question}

Passages:
{passages}
"""
