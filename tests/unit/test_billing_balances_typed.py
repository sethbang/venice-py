"""``BillingBalances`` types the credit balances the spec declares."""

from venice_ai.types.api.billing import BillingBalances


def test_credit_balances_are_declared_fields() -> None:
    assert "bundledCredits" in BillingBalances.model_fields
    assert "earnedCredits" in BillingBalances.model_fields


def test_credit_balances_parse_as_fields_not_extras() -> None:
    balances = BillingBalances.model_validate(
        {"diem": None, "usd": 25, "bundledCredits": 10, "earnedCredits": 5}
    )
    assert balances.bundledCredits == 10.0
    assert balances.earnedCredits == 5.0
    assert not balances.model_extra


def test_credit_balances_are_optional_and_nullable() -> None:
    assert BillingBalances().bundledCredits is None
    assert BillingBalances().earnedCredits is None
    assert BillingBalances.model_validate({"bundledCredits": None}).bundledCredits is None


def test_an_unknown_balance_is_still_preserved() -> None:
    balances = BillingBalances.model_validate({"futureCredits": 1})
    assert balances.model_extra == {"futureCredits": 1}
