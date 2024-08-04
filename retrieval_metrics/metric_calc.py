"""Generates a csv file containing the metrics for a prompt in separate columns.
"""

import csv
import math  # for logarithmic discount
import string

import nltk  # for word tokenization
from nltk.corpus import stopwords

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


def string_format(sentence: str) -> str:
    """Removes punctuation from the string and makes all characters lowercase."""
    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans("", "", string.punctuation + "•"))
    return sentence


def get_keywords(query: str) -> set:
    """Takes a query and extracts keywords to put in a separate set."""
    # format and tokenize the query
    query = string_format(query)
    words = nltk.word_tokenize(query)  # returns a list

    # remove common stop words from the list
    common_words = set(stopwords.words("english")).union({"•"})
    keywords = set()
    for word in words:
        # skip over common words
        if word in common_words:
            continue

        # add keywords to a new set
        keywords.add(word)
    return keywords


def get_source_relevance(source: str, keywords: set) -> int:
    """Returns 1 if a source is relevant, 0 otherwise based on searching for keywords."""
    # format and tokenize the source
    source = string_format(source)
    source_words = nltk.word_tokenize(source)

    match_count = 0  # to keep track of how many keywords are in a source_words

    # search for keywords in the source
    for word in source_words:
        if word in keywords:
            match_count += 1
            continue

        # if more than 3 key words are in a source it is relevant_count
        if match_count > 3:
            return 1
    return 0


def get_total_relevance(sources: list, keywords: set, k: int) -> int:
    """Returns an int representing the number of relevant sources out of the top k sources."""
    relevance = 0
    # sum up the relevances
    for source in sources[1 : k + 1]:
        relevance += get_source_relevance(source, keywords)
    return relevance


def precision_at_k(sources: list, keywords: set, k: int) -> float:
    """Returns the average percent of the retrieved resources that were relevant out of k sources for one q/a pair."""
    # evaluate the relevance of the sources
    relevance = get_total_relevance(sources, keywords, k)

    # calculate the precision@k
    precision = relevance / k
    return round(precision, 3)


def average_precision(sources: list, keywords: set, k: int) -> float:
    """Returns the average precision of a q/a pair for the top-k sources."""
    total_avgs = 0

    # calculate average precisions for each k
    for i in range(k):
        source = sources[i]
        total_avgs += precision_at_k(sources, keywords, k) * get_source_relevance(
            source, keywords
        )

    # avoid zero division
    total_rel = get_total_relevance(sources, keywords, k)
    if total_rel == 0:
        ap = 0
    # calculate the average precision
    else:
        ap = round(total_avgs / total_rel, 3)
    return ap


def mean_reciprocal_rank(sources: list, keywords: set) -> float:
    """Returns the mean reciprocal rank of a q/a pair of the sources."""
    # variables needed for calculation
    total_q = len(sources) - 1
    total_rr = 0  # total the reciprocal ranks

    # calculate the reciprocal rank
    for i in range(total_q):
        i += 1  # prevent zero division by indexing from 1
        relevance = get_source_relevance(sources[i], keywords)
        if relevance == 1:
            total_rr += 1 / i
    return round(total_rr / total_q, 3)


def cumulative_gain(relevances: list) -> float:
    """Returns the total sum of relevancy for one query/documents pair with min being 0 (no relevancy)
    and max being k^2 (extremely relevant)
    """
    cg = 0
    for relevance in relevances:
        cg += int(relevance)
    return cg


def normalized_discounted_cg(relevances: list) -> float:
    """Return the normalized discounted cumulative gain based on the relevances provided."""
    dcg = 0  # non normalized
    for i, relevance in enumerate(relevances):
        dcg += int(relevances[i]) / math.log2(
            i + 2
        )  # plus 2 since we use zero indexing

    # normalize the metric
    ideal_dcg = len(relevances) ^ 2
    ndcg = dcg / ideal_dcg

    return round(ndcg, 3)


def generate_metrics(in_f: str, out_f: str, k: int):
    """Generates a csv file contains the prompts and the metrics for their retrieved
    documents. Returns the name of the output file.
    """
    # write the metrics to a csv file
    with open(out_f, mode="w", newline="") as f_out:
        out_f = csv.writer(f_out)

        # write the header
        out_f.writerow(
            [
                "prompt",
                "precision at k",
                "mean reciprocal rank",
                "average precision",
                "cumulative gain",
                "normalized discounted cumulative gain",
            ]
        )

        # read in the queries and responses from the csv file
        with open(in_f, mode="r", newline="") as f_in:
            responses = csv.reader(f_in)
            # skip the header
            next(responses)

            # get the metrics and write to the out file
            for response in responses:
                query = response[0]
                sources = response[1 : k + 1]
                str_relevances = response[k + 1 : 2 * k + 2]

                # convert relevances to ints
                relevances = []
                for relevance in str_relevances:
                    if relevance == "":
                        relevances.append(0)
                    else:
                        relevances.append(int(relevance))

                # get the keywords from the prompts
                keywords = get_keywords(query)

                p_at_k = precision_at_k(sources, keywords, k)
                mrr = mean_reciprocal_rank(sources, keywords)
                ap = average_precision(sources, keywords, k)
                cg = cumulative_gain(relevances)
                ndcg = normalized_discounted_cg(relevances)
                out_f.writerow([query, p_at_k, mrr, ap, cg, ndcg])
    return out_f


def individual_relevances(in_f: str, out_f: str):
    """Writes a csv file with determined relevances. 1 for relevant and 0 for non relevant."""
    # write the relevances to a csv file
    with open(out_f, mode="w", newline="") as f_out:
        out_f = csv.writer(f_out)
        out_f.writerow(
            ["relevance1", "relevance2", "relevance3", "relevance4", "relevance5"]
        )

        with open(in_f, mode="r", newline="") as f_in:
            responses = csv.reader(f_in)
            # skip the header
            next(responses)

            # get the relevances for each source
            for response in responses:
                keywords = get_keywords(response[0])
                relevances = []
                for source in response[1:6]:  # first column is the query
                    relevances.append(get_source_relevance(source, keywords))
                out_f.writerow(relevances)
    return out_f


def binarize_data(in_f: str, out_f: str):
    """Binarizes the human generated relevances. If the score is a 3 or 4 then it is relevant (1),
    otherwise it isn't (0). Saves the binary data to a csv file.
    """
    with open(out_f, mode="w", newline="") as f_out:
        out_f = csv.writer(f_out)
        out_f.writerow(
            ["relevance1", "relevance2", "relevance3", "relevance4", "relevance5"]
        )

        with open(in_f, mode="r", newline="") as f_in:
            responses = csv.reader(f_in)
            # skip the header
            next(responses)
            for response in responses:
                relevances = []
                for relevance in response[6:]:  # only look at the metrics
                    if relevance == "3" or relevance == "4":
                        relevances.append(1)
                    else:
                        relevances.append(0)
                out_f.writerow(relevances)
    return out_f


if __name__ == "__main__":
    k = 5
    generate_metrics("responses.csv", "metrics.csv", k)
    individual_relevances("retrieved_sources.csv", "binary_relevance.csv")
    binarize_data("retrieved_sources.csv", "binarized_human_metrics.csv")
