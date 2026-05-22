"""Risk classification based on ABCDE morphological scores.

Takes the combined ABCDE total score and maps it to a clinical risk
level that a healthcare professional can quickly interpret.
"""

from __future__ import annotations

from dataclasses import dataclass

from medsight.abcde import ABCDEResult


@dataclass
class RiskAssessment:
    """The clinical risk level derived from ABCDE analysis."""

    level: str           # "Low", "Moderate", "High", or "Refer"
    total_score: float   # the raw ABCDE total
    summary: str         # human-readable explanation


# ── Thresholds ───────────────────────────────────────────
#
#   Total ABCDE score ranges (max possible = 2+2+3+2+0 = 9):
#     0–2   → Low risk
#     3–4   → Moderate risk
#     5–6   → High risk
#     7+    → Refer for biopsy
#
# These are conservative thresholds.  In a real clinical system
# they would be calibrated against validated datasets.


def classify_risk(abcde: ABCDEResult) -> RiskAssessment:
    """Map an ABCDE result to a risk level.

    Args:
        abcde: The morphological analysis result.

    Returns:
        A RiskAssessment with the level, score, and a plain-language
        summary explaining why.
    """
    from medsight.config import (
        RISK_THRESHOLD_LOW, RISK_THRESHOLD_MODERATE, RISK_THRESHOLD_HIGH,
    )

    score = abcde.total_score

    if score <= RISK_THRESHOLD_LOW:
        level = "Low"
        summary = _build_summary(abcde, "Low-risk lesion. Regular monitoring recommended.")

    elif score <= RISK_THRESHOLD_MODERATE:
        level = "Moderate"
        summary = _build_summary(abcde, "Moderate-risk features detected. Clinical follow-up advised.")

    elif score <= RISK_THRESHOLD_HIGH:
        level = "High"
        summary = _build_summary(abcde, "High-risk morphological features. Dermatological evaluation recommended.")

    else:
        level = "Refer"
        summary = _build_summary(abcde, "Multiple high-risk criteria met. Referral for biopsy strongly recommended.")

    return RiskAssessment(
        level=level,
        total_score=round(score, 1),
        summary=summary,
    )


def _build_summary(abcde: ABCDEResult, conclusion: str) -> str:
    """Create a readable summary listing which criteria flagged."""
    parts: list[str] = []

    if abcde.asymmetry_score >= 1.0:
        parts.append(f"asymmetry ({abcde.asymmetry_score:.0f}/2)")
    if abcde.border_score >= 1.0:
        parts.append(f"irregular border ({abcde.border_score:.0f}/2)")
    if abcde.color_score >= 1.0:
        parts.append(f"{abcde.color_count} distinct colors ({abcde.color_score:.0f}/3)")
    if abcde.diameter_score >= 1.0:
        parts.append(f"diameter {abcde.diameter_mm:.1f}mm ({abcde.diameter_score:.0f}/2)")

    if parts:
        flags = "Flagged criteria: " + ", ".join(parts) + ". "
    else:
        flags = "No criteria flagged. "

    return flags + conclusion
