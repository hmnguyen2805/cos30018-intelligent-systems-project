"""
Reading the tags Detection puts in `detector_notes`.

Detection's category model tags its notes like "[category=BruteForce] ...".
Both the Mitigation Manager (to choose techniques) and the Judge (to check
consistency) read this tag, so the parsing lives here, in one place.
"""
import re
from typing import Optional

_CATEGORY_TAG = re.compile(r"\[category=([A-Za-z]+)\]")


def parse_category(detector_notes: Optional[str]) -> Optional[str]:
    """Return the attack category Detection tagged in its notes, or None."""
    if not detector_notes:
        return None
    match = _CATEGORY_TAG.search(detector_notes)
    return match.group(1) if match else None


def strip_category_tag(detector_notes: Optional[str]) -> str:
    """Return the notes text without the [category=X] tag ("" if nothing is left)."""
    if not detector_notes:
        return ""
    return _CATEGORY_TAG.sub("", detector_notes).strip()
