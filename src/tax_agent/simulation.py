"""Synthetic taxpayers: a year of transactions with the true category of every row.

No real data is involved: the merchants, the amounts and the people are invented, so this can live in
a public repository. A simulated user has a salary, a household status and a year of `Transaction`
rows whose category is the truth. `SimUser.unlabeled()` is what an export would contain (no category):
something has to work out the category from the merchant name, and the truth says how well it did.

Some merchants have a clear name ("... 약국") and some have a name that says nothing ("봄날 건강센터"):
`obscure_rate` is the share of special-category purchases that use the second kind. That is what makes
the classification question worth measuring.

Simplifications, on purpose: every medical payment is for the taxpayer ("self"); a medical payment
counts only as medical, not also as card usage (see the open questions in transactions.py); the
excluded kinds (insurance, utilities, a car, gift certificates) follow common knowledge and still need
checking against the NTS guide.
"""
import random
from dataclasses import dataclass, replace
from datetime import date, timedelta

from tax_agent.datasource import UserProfile
from tax_agent.transactions import Transaction

# (clear names, obscure names). "{n}" becomes a number so that the same shop is not always the same row.
MERCHANTS: dict[str, tuple[list[str], list[str]]] = {
    "general": (["한솔마트 {n}호점", "푸른식당", "달빛카페", "온라인몰 하늘", "맛나치킨", "24편의점 {n}호점", "바른옷가게", "초록주유소"], []),
    "medical": (["행복내과의원", "새봄약국", "미소치과", "한마음안과", "청담한의원"],
                ["윤샘 마취통증클리닉", "다나은 메디케어", "봄날 건강센터", "웃는얼굴 교정센터"]),
    "transit": (["지하철 승차", "시내버스 요금", "광역버스 정산", "교통카드 충전"],
                ["하늘길 이동권 충전", "도시이동 정산", "환승 정기권"]),
    "market": (["남문시장 정육점", "중앙시장 청과", "서문시장 떡집", "동부시장 수산"],
               ["정다운 반찬가게", "할매 순대국 1번 점포", "푸른청과 12번"]),
    "culture": (["북스토리 서점", "시네마하늘 영화관", "한빛수영장", "예술의터 공연장", "파워헬스장"],
                ["책방 이음", "하루극장", "물결스포츠센터", "아트홀 소극장"]),
    "excluded": (["하나안심 보험료", "아파트 관리비", "한빛 전기요금", "도시가스 요금", "푸른자동차 차량구입", "백화점 상품권 구입"],
                 ["해든생명 자동이체", "안심보장 납입", "우리동네 관리사무소", "마이카 할부금"]),
    "rent": (["월세 이체 (가상임대인)"], ["집주인 송금"]),
}
METHOD_WEIGHTS = {  # credit, debit, cash_receipt
    "general": (75, 20, 5), "medical": (60, 30, 10), "transit": (60, 40, 0), "market": (50, 30, 20),
    "culture": (60, 30, 10), "excluded": (85, 15, 0),
}
PER_MONTH = {"general": (12, 20), "medical": (0, 2), "transit": (2, 5), "market": (0, 3), "culture": (0, 2), "excluded": (2, 4)}


@dataclass(frozen=True)
class SimUser:
    seed: int
    profile: UserProfile
    transactions: list[Transaction]  # with the true category

    def unlabeled(self) -> list[Transaction]:
        """The rows as an export would give them: the category is unknown."""
        return [replace(t, category="unclassified") for t in self.transactions]


def _amount(rng: random.Random, category: str) -> int:
    if category == "general":
        base = rng.choice((9_000, 15_000, 28_000, 45_000, 80_000, 150_000)) * rng.uniform(0.6, 1.4)
    elif category == "medical":
        base = rng.choice((12_000, 30_000, 85_000, 240_000)) * rng.uniform(0.7, 1.3)
    elif category == "transit":
        base = rng.choice((20_000, 30_000, 50_000)) * rng.uniform(0.8, 1.2)
    elif category == "market":
        base = rng.choice((8_000, 20_000, 45_000)) * rng.uniform(0.7, 1.3)
    elif category == "culture":
        base = rng.choice((15_000, 30_000, 60_000)) * rng.uniform(0.7, 1.3)
    else:  # excluded: insurance, utilities and so on
        base = rng.choice((40_000, 90_000, 150_000, 400_000)) * rng.uniform(0.8, 1.2)
    return max(int(round(base, -3)), 1_000)


def _merchant(rng: random.Random, category: str, obscure_rate: float) -> str:
    clear, obscure = MERCHANTS[category]
    pool = obscure if obscure and rng.random() < obscure_rate else clear
    return rng.choice(pool).format(n=rng.randint(1, 40))


def simulate_user(seed: int, year: int = 2025, obscure_rate: float = 0.15, refund_rate: float = 0.04) -> SimUser:
    """One synthetic taxpayer. The same seed gives the same user."""
    rng = random.Random(seed)
    salary = rng.randrange(30_000_000, 90_000_000, 100_000)
    homeless = rng.random() < 0.7
    rent_monthly = rng.choice((0, 400_000, 550_000, 700_000)) if homeless else 0
    profile = UserProfile(f"sim_{seed:04d}", salary, homeless)
    card_share = rng.uniform(0.25, 0.5)  # general spending as a share of the salary

    rows: list[Transaction] = []
    for month in range(1, 13):
        for category, (low, high) in PER_MONTH.items():
            for _ in range(rng.randint(low, high)):
                day = date(year, month, rng.randint(1, 28))
                credit, debit, cash = METHOD_WEIGHTS[category]
                method = rng.choices(("credit", "debit", "cash_receipt"), (credit, debit, cash))[0]
                rows.append(Transaction(day.isoformat(), _amount(rng, category), _merchant(rng, category, obscure_rate),
                                        method, category))
        if rent_monthly:
            rows.append(Transaction(date(year, month, 5).isoformat(), rent_monthly,
                                    _merchant(rng, "rent", obscure_rate), "other", "rent"))

    general_total = sum(t.amount for t in rows if t.category == "general")
    scale = card_share * salary / general_total if general_total else 1.0
    rows = [replace(t, amount=max(int(round(t.amount * scale, -3)), 1_000)) if t.category == "general" else t for t in rows]

    refunds = []
    for t in rows:  # a cancellation some days later, for the whole amount or a part of it
        if t.category in ("general", "medical", "excluded") and rng.random() < refund_rate:
            later = date.fromisoformat(t.date) + timedelta(days=rng.randint(1, 10))
            if later.year == year:
                part = t.amount if rng.random() < 0.7 else max(t.amount // 2 // 1000 * 1000, 1_000)
                refunds.append(replace(t, date=later.isoformat(), amount=-part))
    rows = sorted(rows + refunds, key=lambda t: (t.date, t.merchant))
    return SimUser(seed, profile, rows)
