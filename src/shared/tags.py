"""
Reading the tags Detection puts in `detector_notes`.

Detection tags its notes like "[category=DoS] [label=DoS Hulk] Single flow ...".
The Mitigation Manager (to choose techniques), the Judge (to check consistency)
and the UI read these tags, so the parsing lives here, in one place.

    parse_category(notes)  -> "DoS"        coarse category (one word)
    parse_label(notes)     -> "DoS Hulk"   fine CICIDS2017 label (may contain spaces)
    strip_tags(notes)      -> the notes text with every [key=value] tag removed

If DetectionResult has the structured fields (attack_category, attack_label),
prefer those; these helpers are the fallback that works on the notes alone.
"""
import re
from typing import Optional

_CATEGORY_TAG = re.compile(r"\[category=([A-Za-z]+)\]")
# Labels can contain spaces and hyphens, e.g. "Web Attack - XSS", so match up to the "]".
_LABEL_TAG = re.compile(r"\[label=([^\]]+)\]")
# Any "[key=value]" tag Detection adds, now or later.
_ANY_TAG = re.compile(r"\[[a-z_]+=[^\]]*\]")


def parse_category(detector_notes: Optional[str]) -> Optional[str]:
    """Return the attack category Detection tagged in its notes, or None."""
    if not detector_notes:
        return None
    match = _CATEGORY_TAG.search(detector_notes)
    return match.group(1) if match else None


def parse_label(detector_notes: Optional[str]) -> Optional[str]:
    """Return the fine CICIDS2017 label Detection tagged in its notes, or None."""
    if not detector_notes:
        return None
    match = _LABEL_TAG.search(detector_notes)
    return match.group(1).strip() if match else None


def strip_tags(detector_notes: Optional[str]) -> str:
    """Return the notes text without any [key=value] tags ("" if nothing is left).

    Used before embedding search: the tags are labels, not a description of the
    traffic, so they shouldn't steer which technique the search matches."""
    if not detector_notes:
        return ""
    return " ".join(_ANY_TAG.sub("", detector_notes).split())


def strip_category_tag(detector_notes: Optional[str]) -> str:
    """Kept for existing callers: now strips every tag, see strip_tags."""
    return strip_tags(detector_notes)
