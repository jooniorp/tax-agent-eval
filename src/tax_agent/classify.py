"""Work out a transaction's deduction category from what an export contains (merchant, method).

A classifier is any function from a list of transactions to the same list with `category` filled in.
The rule-based one below is the baseline: keywords in the merchant name. It is cheap and deterministic
and it fails exactly where the name says nothing (a "건강센터" that is a clinic), which is what the
simulator measures and what a model-based classifier has to beat.
"""
import json
import re
from collections.abc import Callable
from dataclasses import replace

from langchain_core.messages import HumanMessage, SystemMessage

from tax_agent.transactions import Transaction

Classifier = Callable[[list[Transaction]], list[Transaction]]

# Checked in this order: the first match wins ("시내버스 요금" is transit before "요금" could make it excluded).
KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("rent", ("월세", "임대")),
    ("medical", ("의원", "병원", "약국", "치과", "한의원", "안과")),
    ("transit", ("지하철", "버스", "교통카드")),
    ("market", ("시장",)),
    ("culture", ("서점", "영화", "수영", "공연", "헬스", "박물관", "미술관")),
    ("excluded", ("보험", "관리비", "요금", "차량구입", "상품권")),
)


def classify_one(t: Transaction) -> Transaction:
    """A keyword match decides. With no match, a card or cash-receipt payment is ordinary spending, and a
    transfer stays unclassified: nothing says what it was for, and guessing would change a deduction."""
    for category, words in KEYWORDS:
        if any(w in t.merchant for w in words):
            return replace(t, category=category)
    return replace(t, category="general" if t.method != "other" else "unclassified")


def classify_rules(transactions: list[Transaction]) -> list[Transaction]:
    return [classify_one(t) for t in transactions]


# --- a model as the classifier ------------------------------------------------------------------

CATEGORY_HELP = """You label Korean card and bank transactions for the year-end tax settlement of a salaried worker.
For each numbered line (merchant | payment method) answer one category:
- general: ordinary eligible spending (groceries, restaurants, cafes, online shopping, clothes, fuel, convenience stores)
- market: stalls and shops in a traditional market
- transit: public transit fares and transit card top-ups
- culture: books, performances, movies, museums, sports facilities (gyms, swimming pools)
- medical: hospitals, clinics, pharmacies, dental and oriental medicine
- rent: monthly rent paid to a landlord
- excluded: not eligible for the card deduction: insurance premiums, utilities and management fees, car purchase or car installments, gift certificates, tuition
- unclassified: you cannot tell from the name
Use only what the merchant name and the payment method tell you. When you cannot tell, answer unclassified: a wrong guess changes someone's tax.
Reply with one JSON object that maps each line number (as a string) to its category, for example {"1": "general", "2": "medical"}, and nothing else."""

CATEGORY_CHOICES = ("general", "market", "transit", "culture", "medical", "rent", "excluded", "unclassified")
CHUNK = 50  # lines per request


def _key(t: Transaction) -> tuple[str, str]:
    """What the model is asked about: the merchant without its numbers ("한솔마트 12호점" is one shop kind), and the method."""
    return re.sub(r"\d+", "", t.merchant).strip(), t.method


class LLMClassifier:
    """Labels each distinct (merchant, method) once with a chat model and remembers the answer, so a thousand
    rows cost a few lines of prompt. Counts the tokens it used."""

    def __init__(self, model):
        self.model = model
        self.cache: dict[tuple[str, str], str] = {}
        self.tokens_in = self.tokens_out = 0

    def _ask(self, keys: list[tuple[str, str]]) -> list[str]:
        lines = "\n".join(f"{i}. {merchant} | {method}" for i, (merchant, method) in enumerate(keys, 1))
        reply = self.model.invoke([SystemMessage(CATEGORY_HELP), HumanMessage(lines)])
        usage = getattr(reply, "usage_metadata", None) or {}
        self.tokens_in += usage.get("input_tokens", 0)
        self.tokens_out += usage.get("output_tokens", 0)
        match = re.search(r"\{.*\}", reply.text, re.S)
        try:
            answers = json.loads(match.group()) if match else {}
        except json.JSONDecodeError:
            answers = {}
        if not isinstance(answers, dict):
            answers = {}
        # By line number, not by position: a model that skips a line must not shift every answer after it.
        # A missing, unknown or broken answer is "unclassified": guessing would change a deduction.
        return [a if (a := answers.get(str(i))) in CATEGORY_CHOICES else "unclassified" for i in range(1, len(keys) + 1)]

    def __call__(self, transactions: list[Transaction]) -> list[Transaction]:
        todo = sorted({_key(t) for t in transactions} - set(self.cache))
        for start in range(0, len(todo), CHUNK):
            chunk = todo[start:start + CHUNK]
            self.cache.update(zip(chunk, self._ask(chunk)))
        return [replace(t, category=self.cache[_key(t)]) for t in transactions]
