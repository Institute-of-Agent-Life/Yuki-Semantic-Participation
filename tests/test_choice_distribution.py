import pytest
from pydantic import ValidationError

from yuki_participation.models import Choice


def test_two_decimal_rounding_still_sums_to_one():
    low = Choice(choice="a", probabilities={"a": 0.34, "b": 0.33, "c": 0.32})
    high = Choice(choice="a", probabilities={"a": 0.34, "b": 0.34, "c": 0.33})

    assert low.choice == "a"
    assert high.p("a") == 0.34


def test_a_distribution_far_from_one_is_rejected():
    with pytest.raises(ValidationError):
        Choice(choice="a", probabilities={"a": 0.5, "b": 0.4})
