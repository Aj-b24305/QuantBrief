from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class ShockScenario:
    """A named historical stress window replayed against the portfolio."""

    id: str
    name: str
    description: str
    start: date
    end: date


# Fixed historical shock windows (inclusive). Used both to size the download
# window and to compute scenario returns. Add more entries here to extend the
# stress library — the calculator and API pick them up automatically.
SCENARIOS: tuple[ShockScenario, ...] = (
    ShockScenario(
        id="covid_2020",
        name="COVID-19 Crash (2020)",
        description="Peak-to-trough of the Feb-Mar 2020 selloff triggered by the COVID-19 pandemic.",
        start=date(2020, 2, 19),
        end=date(2020, 3, 23),
    ),
    ShockScenario(
        id="rate_shock_2022",
        name="2022 Rate Shock",
        description="Aggressive Fed tightening drove equities to their 2022 trough (Jan 3 - Oct 12).",
        start=date(2022, 1, 3),
        end=date(2022, 10, 12),
    ),
    ShockScenario(
        id="gfc_2008",
        name="Global Financial Crisis (2008)",
        description="Broad equity drawdown around the Lehman collapse and the Nov 2008 trough.",
        start=date(2008, 9, 1),
        end=date(2008, 11, 20),
    ),
)