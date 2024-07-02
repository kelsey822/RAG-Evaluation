import csv
import string
import nltk #for word tokenization
#nltk.download('punkt')

"""Generates a csv file containing the metrics for a prompt in separate columns.
"""

def string_format(sentence):
    """Removes punctuation from the string and makes all characters lowercase.
    """
    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans('', '', string.punctuation + '•'))
    return sentence

def get_keywords(query) -> set:
    """Takes a query and extracts keywords to put in a separate list.
    """
    #format the query
    #query = query.lower()
    #query = query.translate(str.maketrans('', '', string.punctuation))
    query = string_format(query)

    # tokenize the query by word and put the words into a list
    words = nltk.word_tokenize(query) #returns a list

    # remove common stop words from the list
    common_words = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
                    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
                    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
                    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
                    "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has",
                    "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but",
                    "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
                    "about", "against", "between", "into", "through", "during", "before", "after",
                    "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
                    "over", "under", "again", "further", "then", "once", "here", "there", "when",
                    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
                    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
                    "too","very", "s", "t", "can", "will", "just", "don", "should", "now", "boston",
                    "children's", "children", "hospital", "bch", "•"}
    keywords = set()
    for word in words:
        # skip over common words
        if word in common_words:
            continue

        # add keywords to a new set
        keywords.add(word)
    return keywords


def get_relevance(query, sources: list, keywords) -> int:
    """Returns an int representing the number of relevant sources.
    """
    relevance = 0;
    for source in sources:
        # format and tokenize the source
        source = string_format(source)
        source_words = nltk.word_tokenize(source)

        match_count = 0 # to keep track of how many keywords are in a source_words

        # search for keywords in the source
        for word in source_words:
            if word in keywords:
                match_count += 1;
            continue

        # if more than 3 key words are in a source it is relevant_count
        if match_count > 3:
            relevance += 1
    return relevance


def precision_at_k(row, k) -> float:
    """Returns the average percent of retrieved resources that were relevant for one q/a pair
    Precondtions: the row contains a queries and at least one source, 0 < k < 7
    """

    #get the keywords from the prompt
    query = row[0]
    keywords = get_keywords(query)

    #evalute the relevance of the sources
    sources = row[1:k+1]
    relevance = get_relevance(query, sources, keywords)

    #calculate the precision@k
    precision = relevance / len(sources)
    return precision


def generate_metrics(input, output, k):
    """ Generates a csv file contains the prompts and the metrics for their retrieved
    documents.
    """
    # write the metrics to a csv file
    with open(output, mode = 'w', newline = '') as f_out:
        output = csv.writer(f_out)

        # write the header
        output.writerow(["prompt", "precision_at_k"])

        # read in the queries and responses from the csv file
        with open(input, mode = 'r', newline = '') as f_in:
            responses = csv.reader(f_in)
            # skip the header
            next(responses)

            #get the metrics and write to the out file
            for response in responses:
                query = response[0]
                sources = response[1:k+1]
                p_at_k = precision_at_k(response, k)
                output.writerow([query, p_at_k])

    print("metrics generated")

if __name__ == "__main__":
    k = 5
    generate_metrics("responses.csv", "metrics.csv", k)
