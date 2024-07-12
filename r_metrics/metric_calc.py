"""Generates a csv file containing the metrics for a prompt in separate columns.
"""

import csv
import string
import nltk  # for word tokenization
from nltk.corpus import stopwords
import math # for logarithmic discount


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
        cg += relevance
    return cg

def normalized_discounted_cg(relevances: list, k: int) -> float:
    """Return the normalized discounted cumulative gain based on the relevances provided.
    """
    dcg = 0 #non normalized
    for i in range(len(relevances)):
        dcg += relevance[i] / math.log2(i + 1)

    #normalize the metric
    ideal_dcg = 


def generate_metrics(input: str, output: str, k: int):
    """Generates a csv file contains the prompts and the metrics for their retrieved
    documents. Returns the name of the output file.
    """
    # write the metrics to a csv file
    with open(output, mode="w", newline="") as f_out:
        output = csv.writer(f_out)

        # write the header
        output.writerow(
            ["prompt", " precision at k", " mean reciprocal rank", " average precision"]
        )

        # to generate binary metrics
        # read in the queries and responses from the csv file
        with open(input, mode="r", newline="") as f_in:
            responses = csv.reader(f_in)
            # skip the header
            next(responses)

            # get the metrics and write to the out file
            for response in responses:
                query = response[0]
                sources = response[1 : k + 1]

                # get the keywords from the prompts
                keywords = get_keywords(query)

                p_at_k = precision_at_k(sources, keywords, k)
                mrr = mean_reciprocal_rank(sources, keywords)
                ap = average_precision(sources, keywords, k)
                output.writerow([query, p_at_k, mrr, ap])
        print("binary metrics generated")

        # to generate non binary metrics
        # read in queries, responses, and human scored relevances
        with open(input, mode = "r", newline = "") as f_in:
            scored_responses = csv.reader(f_in)
            #skip the heaader
            next(scored_responses)

            #get the relevance scores
            for response in scored_responses:
                relevances = scored_responses[k + 1 : 2k + 2]


        print("nonbinary metrics generated")

    return output


if __name__ == "__main__":
    k = 5
    generate_metrics("responses.csv", "metrics.csv", k)
