"""Question-type router. Routes by QUESTION TEXT, never row position."""

EASY_TYPES = {
    "What trends can be observed": "Revenue",
    "balance sheet reflect": "BalanceSheet",
    "operating, investing and financing cash flow": "CashFlow",
    "R&D ratio": "RnD",
}

EXPERT_TYPES = {
    "top three focuses on revenue": "RevenueFocus",
    "allocating capital": "CapitalAllocation",
    "maintaining profit margins": "MarginStrategy",
    "capital expenditures and their strategic significance": "Capex",
}


def route(question: str) -> str:
    for markers in (EASY_TYPES, EXPERT_TYPES):
        for marker, qtype in markers.items():
            if marker in question:
                return qtype
    raise ValueError(f"unroutable question: {question[:120]!r}")


def tier(question: str) -> str:
    return "easy" if route(question) in set(EASY_TYPES.values()) else "expert"
