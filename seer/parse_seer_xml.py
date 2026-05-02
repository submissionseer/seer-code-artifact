from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class SeerParseResult:
    requirements: list[str]
    present_map: dict[str, list[int]]
    missing: list[str]
    raw_text: str
    valid: bool


_REQ_RE = re.compile(r"^f(\d+)\)\s*(.+)$")
# Matches f1->p2 or f1->p2,p3 or f1->2,3 formats
_MAP_RE = re.compile(r"^f(\d+)->(.+)$")


def _extract_block(text: str, tag: str) -> str:
    start = f"<{tag}>"
    end = f"</{tag}>"
    if start not in text or end not in text:
        return ""
    return text.split(start, 1)[1].split(end, 1)[0].strip()


def parse_seer_xml(text: str) -> SeerParseResult:
    req_block = _extract_block(text, "specific_information_required")
    present_block = _extract_block(text, "present_information")
    missing_block = _extract_block(text, "missing_information")

    requirements: list[str] = []
    present_map: dict[str, list[int]] = {}
    missing: list[str] = []

    for line in filter(None, req_block.splitlines()):
        match = _REQ_RE.match(line.strip())
        if match:
            requirements.append(match.group(2).strip())

    if present_block.strip().upper() != "NONE":
        for line in filter(None, present_block.splitlines()):
            match = _MAP_RE.match(line.strip())
            if match:
                f_id = f"f{match.group(1)}"
                # Handle both "p2,p3" and "2,3" formats
                pid_str = match.group(2)
                pids = []
                for part in pid_str.split(","):
                    part = part.strip()
                    if part.startswith("p"):
                        part = part[1:]
                    if part.isdigit():
                        pids.append(int(part))
                present_map[f_id] = pids

    if missing_block.strip().upper() != "NONE":
        for line in filter(None, missing_block.splitlines()):
            line = line.strip()
            if line and line.upper() != "NONE":
                missing.append(line)

    valid = bool(requirements)
    return SeerParseResult(requirements, present_map, missing, text, valid)
