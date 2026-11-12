"""Numbers recovered from spelled-out narration."""
import pytest

from why_engine.art.numbers import Number, parse_numbers, unit_for

values = lambda t: [n.value for n in parse_numbers(t)]
displays = lambda t: [n.display for n in parse_numbers(t)]


@pytest.mark.parametrize("text,expected", [
    ("twelve pigs", 12), ("twenty five states", 25), ("thirty years", 30),
    ("one hundred", 100), ("two hundred and fifty", 250),
    ("three thousand", 3000), ("one point two", 1.2),
    ("eight", 8), ("nineteen", 19), ("ninety nine", 99),
])
def test_parses_spelled_out_cardinals(text, expected):
    assert values(text) == [pytest.approx(expected)]


@pytest.mark.parametrize("text,expected", [
    ("eighteen seventy seven", 1877), ("nineteen eighty four", 1984),
    ("eighteen eighty", 1880), ("twenty twenty four", 2024),
])
def test_parses_spoken_years(text, expected):
    n = parse_numbers(text)[0]
    assert n.value == expected and n.is_year


def test_a_spoken_year_is_not_read_as_a_sum():
    """'eighteen seventy seven' is 1877, not 18 + 77."""
    assert values("eighteen seventy seven") == [1877]


def test_implausible_year_shapes_fall_back_to_cardinal():
    assert parse_numbers("ninety nine")[0].is_year is False


@pytest.mark.parametrize("text,expected", [
    ("4,000", 4000.0), ("1.5", 1.5), ("25", 25.0), ("1880", 1880.0),
])
def test_parses_digits(text, expected):
    assert values(text) == [pytest.approx(expected)]


def test_four_digit_year_survives_a_sentence_ending_period():
    """A greedy decimal pattern swallows the period and breaks the year test."""
    n = parse_numbers("It happened in 1880.")[0]
    assert n.value == 1880 and n.is_year


def test_a_decimal_is_not_split_at_the_point():
    assert values("It weighs 1.25 kilograms.") == [pytest.approx(1.25)]


def test_attaches_a_following_unit():
    n = parse_numbers("eight kilograms of feed")[0]
    assert n.unit == "KG"


def test_singular_value_takes_a_singular_unit():
    """A card reading '1 ACRES' looks like a bug even when the number is right."""
    assert parse_numbers("one acre")[0].display == "1 ACRE"
    assert parse_numbers("five acres")[0].display == "5 ACRES"


def test_unit_for_returns_none_for_a_non_unit():
    assert unit_for("pigs", 12) is None
    assert unit_for(None, 1) is None


def test_finds_several_numbers_in_one_sentence():
    assert values("eight kilograms against one point two kilograms") == \
        [pytest.approx(8), pytest.approx(1.2)]


def test_year_and_count_in_one_sentence():
    ns = parse_numbers("In eighteen seventy seven they shipped to twenty five states.")
    assert [n.value for n in ns] == [1877, 25]
    assert [n.is_year for n in ns] == [True, False]


def test_sentence_with_no_numbers():
    assert parse_numbers("The fish had not changed.") == []


def test_display_omits_a_trailing_zero():
    assert Number(8.0, "KG", "eight").display == "8 KG"


def test_display_separates_thousands_but_not_years():
    assert Number(4000.0, None, "4000").display == "4,000"
    assert Number(1877.0, None, "1877", is_year=True).display == "1877"


def test_word_that_is_not_a_number_is_ignored():
    assert parse_numbers("one of the states") == [pytest.approx(1)] or \
           values("one of the states") == [1.0]


@pytest.mark.parametrize("text,expected", [
    ("twenty five states", 25),      # second group opens with a unit -> cardinal
    ("twenty twenty four", 2024),    # second group opens with a tens  -> year
    ("eighteen hundred", 1800),
    ("eighteen thousand", 18000),    # a scale word, not a year
])
def test_year_and_cardinal_are_told_apart_by_the_second_group(text, expected):
    assert values(text) == [pytest.approx(expected)]


def test_twenty_five_is_not_read_as_a_year():
    assert parse_numbers("twenty five states")[0].is_year is False


def test_twenty_twenty_four_is_read_as_a_year():
    assert parse_numbers("twenty twenty four")[0].is_year is True
