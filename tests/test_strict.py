from why_engine.grounding.strict import check_verbatim, hard_fact_tokens
from why_engine.models import Claim


def test_extracts_years():
    assert "1877" in hard_fact_tokens("The Commission acted in 1877.")


def test_extracts_plain_numbers_and_percentages():
    toks = hard_fact_tokens("Roughly 25 states and 40% of districts took part.")
    assert "25" in toks
    assert "40" in toks


def test_extracts_capitalised_proper_nouns_midsentence():
    toks = hard_fact_tokens("The program was run by Spencer Baird in Washington.")
    assert "Spencer" in toks
    assert "Baird" in toks
    assert "Washington" in toks


def test_ignores_the_leading_word_of_a_sentence():
    """Sentence-initial capitals are grammar, not proper nouns."""
    assert "Carp" not in hard_fact_tokens("Carp were distributed widely.")


def test_ignores_common_stopwords_even_when_capitalised():
    assert "The" not in hard_fact_tokens("The Commission met. The vote passed.")


def test_check_verbatim_returns_empty_when_all_tokens_present():
    c = Claim(
        text="Spencer Baird led the program in 1877.",
        source_id="s", locator="p.1",
        quote="In 1877 Spencer Baird led the Commission's carp program.",
    )
    assert check_verbatim(c) == []


def test_check_verbatim_flags_a_number_absent_from_the_quote():
    c = Claim(
        text="The program reached 298 districts.",
        source_id="s", locator="p.1",
        quote="The program reached many districts.",
    )
    assert check_verbatim(c) == ["298"]


def test_check_verbatim_flags_a_shifted_year():
    c = Claim(
        text="Hessel imported carp in 1897.",
        source_id="s", locator="p.1",
        quote="Hessel imported carp in 1877.",
    )
    assert check_verbatim(c) == ["1897"]


def test_check_verbatim_flags_a_number_substring_match():
    """40 must not match 400 — word boundaries required."""
    c = Claim(
        text="The program reached 40 districts.",
        source_id="s", locator="p.1",
        quote="The program reached 400 districts.",
    )
    assert check_verbatim(c) == ["40"]


def test_check_verbatim_flags_a_proper_noun_substring_match():
    """Ann must not match Annapolis — word boundaries required."""
    c = Claim(
        text="The director was Ann.",
        source_id="s", locator="p.1",
        quote="The director was Annapolis.",
    )
    assert check_verbatim(c) == ["Ann"]


def test_check_verbatim_passes_a_decimal_genuinely_present():
    """Word boundary match must not break decimals."""
    c = Claim(
        text="The rate was 40.5%.",
        source_id="s", locator="p.1",
        quote="The rate was 40.5% across all districts.",
    )
    assert check_verbatim(c) == []


def test_hard_fact_tokens_preserves_proper_nouns_after_initial():
    """Proper nouns after an initial (e.g. F. Baird) must not be discarded as sentence-initial."""
    toks = hard_fact_tokens("Spencer F. Baird led it.")
    assert "Baird" in toks


def test_hard_fact_tokens_preserves_proper_nouns_after_title():
    """Proper nouns after a title (e.g. Dr. Baird) must not be discarded as sentence-initial."""
    toks = hard_fact_tokens("Dr. Baird led it.")
    assert "Baird" in toks


def test_hard_fact_tokens_drops_bare_initials_from_a_dotted_acronym():
    """'U.S.' must not yield the junk single-character tokens 'U' and 'S'."""
    toks = hard_fact_tokens("The U.S. Congress acted in 1877.")
    assert "Congress" in toks
    assert "1877" in toks
    assert "U" not in toks
    assert "S" not in toks


def test_check_verbatim_matches_proper_nouns_case_insensitively():
    """Scanned government reports routinely extract as all-caps headings."""
    c = Claim(
        text="Baird County",
        source_id="s", locator="p.1",
        quote="THE PROGRAM REACHED BAIRD COUNTY.",
    )
    assert check_verbatim(c) == []


def test_check_verbatim_flags_a_swapped_surname_after_initial():
    """The headline case: a fabricated attribution with an initial should be caught."""
    c = Claim(
        text="Spencer F. Baird led it.",
        source_id="s", locator="p.1",
        quote="Spencer F. Hessel led it.",
    )
    assert check_verbatim(c) == ["Baird"]


def test_a_number_with_a_variable_suffix_is_found():
    """Real rejection from the school-bus run. The CFR writes forces as "3,114W
    newtons", W being the seat's occupancy. \\b after the digits demands a
    non-word character, and 'W' is a word character, so the checker declared a
    figure invented that is sitting in the quote."""
    claim = Claim(text="The seat back is loaded with a force of 3,114 newtons times W.",
                  source_id="cfr-571.222", locator="c7",
                  quote="Apply a force of 3,114W newtons horizontally in the forward direction.")
    assert check_verbatim(claim) == []


def test_a_number_inside_a_longer_number_is_still_a_miss():
    """The relaxed boundary must not go so far that 452 matches inside 14520."""
    claim = Claim(text="It absorbs 452 joules.", source_id="s", locator="c0",
                  quote="It absorbs 14520 joules.")
    assert check_verbatim(claim) == ["452"]


def test_the_standard_number_counts_as_grounded_by_its_own_citation():
    """Also from the run. A claim about "Standard 222" cites a passage of
    571.222, and the section number is not repeated inside every paragraph of
    its own text. The citation is evidence, not decoration."""
    claim = Claim(text="Standard 222's purpose is to reduce deaths.",
                  source_id="cfr-571.222", locator="c0",
                  quote="The purpose of this standard is to reduce the number of deaths.")
    assert check_verbatim(claim) == []


def test_an_invented_figure_is_still_caught_after_both_relaxations():
    """The check has to keep doing the one job it was kept for."""
    claim = Claim(text="It reached 298 districts in 1877.", source_id="cfr-571.222",
                  locator="c0", quote="It reached many districts.")
    assert sorted(check_verbatim(claim)) == ["1877", "298"]


def test_a_number_with_a_letter_suffix_is_extracted_whole():
    """From the run: "3,114W newtons" yielded the token "3". The number pattern
    required a word boundary after the digits, and W is a word character, so it
    backtracked until it found one -- after the first digit. The claim was then
    checked for a bare "3", which is in no quote, and a faithful sentence was
    rejected for inventing a figure it had copied correctly."""
    assert hard_fact_tokens("A force of 3,114W newtons is applied.") == ["3114"]


def test_a_trailing_full_stop_is_not_part_of_the_number():
    assert hard_fact_tokens("It was built in 1877.") == ["1877"]


def test_an_abbreviated_title_is_not_a_hard_fact():
    """Also from the run. "Standard No. 217" made "No" a proper noun to verify,
    and the regulation's own text says "this standard", so every claim naming a
    standard failed. An abbreviation is not a checkable fact; the number after
    it is, and that is checked."""
    assert "No" not in hard_fact_tokens("The purpose of Standard No. 217 is safety.")


def test_the_number_after_the_abbreviation_is_still_checked():
    assert "217" in hard_fact_tokens("The purpose of Standard No. 217 is safety.")
