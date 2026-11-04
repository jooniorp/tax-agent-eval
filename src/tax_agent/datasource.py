"""The boundary between the agent tools and wherever taxpayer data comes from.

Tools only talk to `DataSource`. `MockDataSource` is the only implementation
today; a real-data source can replace it later without touching the tools
(see docs/decisions.md, 2026-10-28).
"""
from dataclasses import dataclass
from typing import Protocol

from tax_agent.mock_api.deduction import CardSpending


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    gross_salary: int  # total annual salary in KRW


class DataSource(Protocol):
    def get_user_profile(self, user_id: str) -> UserProfile: ...

    def get_card_spending(self, user_id: str, year: int) -> CardSpending: ...
