"""Testing for calculating the retrieval metrics.
"""
import pytest

from rag_evaluation.calculate_retrieval_metrics import (
    average_precision, cumulative_gain, get_keywords, get_source_relevance,
    mean_reciprocal_rank, normalized_discounted_cg, precision_at_k,
    string_format)


def test_empty_string_format():
    actual_string = string_format("")
    expected_string = ""
    assert actual_string == expected_string


def test_simple_string_format():
    actual_string = string_format(
        "This, is a simple STRING that needs. to be formatted"
    )
    expected_string = "this is a simple string that needs to be formatted"
    assert actual_string == expected_string


def test_complex_string_format():
    actual_string = string_format("This,!!!!!!????.@@@ string is weird•**&&&&")
    expected_string = "this string is weird"
    assert actual_string == expected_string


def test_empty_keywords():
    assert get_keywords("") == set()


def test_simple_keywords():
    actual = get_keywords("this is a sample of keywords")
    assert actual == {"sample", "keywords"}


def test_complex_keywords():
    actual = get_keywords(
        "this is super duper complex set of keywords with this thing •*"
    )
    assert actual == {"super", "duper", "complex", "set", "keywords", "thing"}


def test_source_relevance_simple_not_relevant():
    assert get_source_relevance("", {"one"}) == 0


def test_source_relevance_simple_almost_relevant():
    assert (
        get_source_relevance("super super super almost relevant source", {"super"}) == 0
    )


def test_source_relevance_simple_just_relevant():
    assert (
        get_source_relevance("super super super super relevant source", {"super"}) == 1
    )


def test_source_relevance_multiple_keywords_not_relevant():
    assert (
        get_source_relevance("one two and there is no more", {"one", "two", "three"})
        == 0
    )


def test_precision_at_k_zero():
    assert precision_at_k(["test"], set(), 1) == 0


def test_precision_at_k_simple():
    sources = [
        "query",
        "super super super super relevant source",
        "source2",
        "source3",
        "source4",
        "source5",
    ]
    keywords = {"super"}
    k = 4
    print(precision_at_k(sources, keywords, k))
    assert precision_at_k(sources, keywords, k) == 0.25


def test_avg_precision_zero():
    assert average_precision(["test"], set(), 1) == 0


def test_avg_precision_simple():
    sources = [
        "query",
        "super super super super relevant source",
        "source2",
        "source3",
        "source4",
        "source5",
    ]
    keywords = {"super"}
    k = 4
    assert average_precision(sources, keywords, k) == 0.25


def test_mean_reciprocal_rank_zero():
    sources = ["query", "source1"]
    keywords = set()
    assert mean_reciprocal_rank(sources, keywords) == 0


def test_mean_reciprocal_rank_simple():
    sources = [
        "query",
        "super super super super relevant source",
        "source2",
        "source3",
        "source4",
        "source5",
    ]
    keywords = {"super"}
    assert mean_reciprocal_rank(sources, keywords) == 0.2


def test_cumulative_gain_zero():
    relevances = []
    assert cumulative_gain(relevances) == 0


def test_cumulative_gain_simple():
    relevances = [0, 1, 2, 3]
    assert cumulative_gain(relevances) == 6


def test_normalized_discounted_cg_gain_zero():
    relevances = []
    assert normalized_discounted_cg(relevances) == 0


def test_normalized_discounted_cg_gain_simple():
    relevances = [0, 1, 2, 3]
    assert normalized_discounted_cg(relevances) == 0.487
