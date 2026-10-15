"""Spoken-English rewriting of numbers, symbols and abbreviations."""

import pytest

from src.text_normalize import (
    normalize_english_text,
    number_to_words,
    ordinal_to_words,
    should_normalize,
    year_to_words,
)


@pytest.mark.parametrize(
    ("number", "words"),
    [
        (0, "zero"),
        (13, "thirteen"),
        (40, "forty"),
        (99, "ninety-nine"),
        (105, "one hundred five"),
        (1200, "one thousand two hundred"),
        (3_000_021, "three million twenty-one"),
    ],
)
def test_number_to_words(number, words):
    assert number_to_words(number) == words


@pytest.mark.parametrize(
    ("number", "words"),
    [
        (1, "first"),
        (2, "second"),
        (3, "third"),
        (12, "twelfth"),
        (20, "twentieth"),
        (23, "twenty-third"),
        (100, "one hundredth"),
    ],
)
def test_ordinal_to_words(number, words):
    assert ordinal_to_words(number) == words


@pytest.mark.parametrize(
    ("year", "words"),
    [
        (1999, "nineteen ninety-nine"),
        (2005, "two thousand five"),
        (2024, "twenty twenty-four"),
        (1900, "nineteen hundred"),
        (1905, "nineteen oh five"),
    ],
)
def test_year_to_words(year, words):
    assert year_to_words(year) == words


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("It costs $5.99.", "It costs five dollars and ninety-nine cents."),
        ("A $1,200 bill", "A one thousand two hundred dollars bill"),
        ("$1 and €3", "one dollar and three euros"),
        ("a $3 million budget", "a three million dollars budget"),
        ("Meet at 3:30 PM", "Meet at three thirty PM"),
        ("at 10:05 or 4:00", "at ten oh five or four o'clock"),
        ("at 9.30am", "at nine thirty AM"),
        ("version 9.30 ships", "version nine point three zero ships"),
        ("up 12.5%", "up twelve point five percent"),
        ("May 1st, 2024", "May first, twenty twenty-four"),
        ("pages 5-10", "pages five to ten"),
        ("Dr. Smith vs. Mr. Jones", "Doctor Smith versus Mister Jones"),
        ("fruit, e.g. apples & pears", "fruit, for example apples and pears"),
        ("ticket #2", "ticket number two"),
        ("an mp3 file", "an mp3 file"),
    ],
)
def test_normalize_english_text(text, spoken):
    assert normalize_english_text(text) == spoken


def test_should_normalize_only_english_scripts():
    assert should_normalize("Hello 5", "English", "qwen")
    assert should_normalize("Hello 5", "auto", "qwen")
    assert not should_normalize("Café à 5 euros", "auto", "qwen")
    assert not should_normalize("Hola 5", "Spanish", "qwen")
    assert should_normalize("Hello 5", "auto", "chatterbox")
