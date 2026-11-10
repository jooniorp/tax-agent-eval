"""The boundary between the agent tools and wherever taxpayer data comes from.

Tools only talk to `DataSource`. `MockDataSource` is the only implementation
today; a real-data source can replace it later without touching the tools
(see docs/decisions.md, 2026-10-28).
"""
from dataclasses import dataclass
from typing import Protocol

from tax_agent.mock_api.deduction import CardSpending
from tax_agent.mock_api.medical import MedicalExpenses
from tax_agent.mock_api.rent import RentPayments


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    gross_salary: int | None  # total annual salary in KRW; None = unknown, the agent must ask
    homeless_household_head: bool | None = None  # head of a household owning no home; None = unknown


class DataSource(Protocol):
    def get_user_profile(self, user_id: str) -> UserProfile: ...

    def get_card_spending(self, user_id: str, year: int) -> CardSpending: ...

    def get_rent_payments(self, user_id: str, year: int) -> RentPayments: ...

    def get_medical_expenses(self, user_id: str, year: int) -> MedicalExpenses: ...
