"""Loads a file of queries and generates responses from policy-chat API.
"""

import csv
import json
import requests
import string


def headers(columns: int) -> list:
    """Creates the header file for a csv file containing the prompt and relevant documents."""
    headers = ["prompt"]
    for i in range(columns):
        headers.append(f"source{i + 1}")
    return headers


def generate_responses(headers: list, input: str, output: str):
    """Save the responses from policy-chat API of a given list of queries to a csv file."""
    # save the query and source texts to a csv file
    with open(output, mode="w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(headers)
        responses = []

        # read in the queries
        with open(input, mode="r") as f_in:
            queries = csv.reader(f_in)
            for row in queries:
                # skip irrelevant rows
                if not row:
                    continue

                # format relevant rows
                query = row[0].strip()
                query = query.translate(str.maketrans("", "", string.punctuation + "•"))
 
                # send POST request
                result = requests.post(
                    "http://localhost:8000/ask", json={"query": query}
                )
                response_data = result.json()

                # get the text from the sources
                sources = response_data.get("source_documents", [])
                source_texts = [
                    json.loads(source["page_content"]).get("text", "")
                    for source in sources
                ]

                # write to the csv file
                full_row = [query] + source_texts + [""]
                writer.writerow(full_row)
                responses.append({"query": query, "response": response_data})

            # also save the prompt and response to a json file
            with open("responses.json", "w") as f:
                json.dump(responses, f, indent=4)

    print("all done")


if __name__ == "__main__":
    columns = 5
    headers = headers(columns)
    generate_responses(headers, "./data/queries.csv", "./data/retrieved_sources.csv")
