"""
Transition marker detection and segmentation.

Implements the segmentation method from "Beyond the Last Answer" (Hammoud et al., 2025).
The reasoning trace is split at linguistic transition markers (e.g. "Wait",
"Alternatively", "Hmm"), enabling regeneration from each marker boundary.
"""

import re
from typing import List


# Transition markers from the BLA paper (Table / Section 3).
# Sorted longest-first so that longer markers match before shorter prefixes
# (e.g. "But wait" before "But", "Let me double-check" before "Let me").
TRANSITION_MARKERS: List[str] = sorted([
    "Wait",
    "Alternatively",
    "Another angle",
    "Another approach",
    "But wait",
    "Hold on",
    "Hmm",
    "Maybe",
    "Looking back",
    "Okay",
    "Let me",
    "First",
    "Then",
    "Alright",
    "Compute",
    "Correct",
    "Good",
    "Got it",
    "I don't see any errors",
    "I think",
    "Let me double-check",
    "Let's see",
    "Now",
    "Remember",
    "Seems solid",
    "Similarly",
    "So",
    "Starting",
    "That's correct",
    "That seems right",
    "Therefore",
    "Thus",
], key=len, reverse=True)

# Build a regex pattern that matches any marker preceded by a newline (or
# start-of-string).  The marker itself is captured so it stays at the
# beginning of its segment.  We use a lookahead-style split: the pattern
# matches the position right before the marker, so `re.split` keeps the
# marker text in the subsequent segment.
#
# Pattern explanation:
#   (?:^|\n)  — start of string or newline (non-capturing)
#   (?=M1|M2|...)  — lookahead for any marker
#
# Because re.split on a zero-width match is tricky, we instead find all
# marker start positions and split manually.
_MARKER_PATTERN = re.compile(
    r"(?:^|\n)\s*(" + "|".join(re.escape(m) for m in TRANSITION_MARKERS) + r")",
    re.IGNORECASE,
)


def segment_at_markers(text: str) -> List[str]:
    """Split *text* into segments at transition marker boundaries.

    Each marker starts a new segment.  The marker text is included at the
    beginning of its segment (except for the first, which contains everything
    before the first marker hit).

    If no markers are found, returns ``[text]``.
    """
    boundaries = get_marker_boundaries(text)
    if not boundaries:
        return [text]

    segments: List[str] = []
    prev = 0
    for pos in boundaries:
        seg = text[prev:pos]
        if seg:  # skip empty leading segment
            segments.append(seg)
        prev = pos

    # Last segment
    tail = text[prev:]
    if tail:
        segments.append(tail)

    # Ensure we always return at least one segment
    return segments if segments else [text]


def get_marker_boundaries(text: str) -> List[int]:
    """Return character positions where each transition marker boundary occurs.

    A boundary is the start of a transition marker that appears at the
    beginning of a line (possibly preceded by whitespace on that line).
    The first boundary is never at position 0 (the very start of the text
    is not a split point).

    Returns a sorted list of unique character offsets.
    """
    positions: List[int] = []
    for m in _MARKER_PATTERN.finditer(text):
        # m.start() points to the newline (or ^); m.start(1) points to
        # the first char of the marker itself.  We want to split at the
        # beginning of the line containing the marker, i.e. right after
        # the preceding newline.
        line_start = m.start()
        if text[line_start] == "\n":
            line_start += 1  # split after the newline
        # Skip if this would be position 0 (no empty first segment)
        if line_start == 0:
            continue
        positions.append(line_start)

    return sorted(set(positions))
