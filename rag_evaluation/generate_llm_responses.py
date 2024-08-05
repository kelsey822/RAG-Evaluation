"""Generates a response from a LLM based on the retrieved documents provided by
policy-chat API and the query.The responses are then saved to a csv file with the
query, retrieved sources, and llm generated response.
"""

import csv
import json
import os
import sys

import requests
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage


def pass_to_policy_chat(query: str):
    """Return a list of the retrieved sources from poliy-chat given a query as an input."""
    # send POST request to policy-chat API
    result = requests.post("http://localhost:8000/ask", json={"query": query})
    response_data = result.json()

    # get the text from the sources
    sources = response_data.get("source_documents", [])
    source_texts = [
        json.loads(source["page_content"]).get("text", "") for source in sources
    ]
    print("sources retrieved")
    return source_texts


def get_llm_response(query: str, sources=None):
    """Takes a command line string query as the argument. Calls policy-chat API to get relevant
    sources and then passes to a LLM for repsonse generation.
    """

    # get the retrieved sources from policy-chat API if needed
    if not sources:
        sources = pass_to_policy_chat(query)

    # format content to pass to the API
    prompt = f""" Based on the following retreieved sources using a policy-chat API, please answer the query.
    Here is the query:
    query = "{query}"
    Here are the retrieved sources, the higher the source number the less relevant the information will be, take that into account.
    Base your response off of the retrieved sources, make them not too long and easily understandable.
    If none of the sources help you answer the question, say that you cannot answer it with the sources. Do not make up an answer.
    source 1: {sources[0]}
    source 2: {sources[1]}
    source 3: {sources[2]}
    source 4: {sources[3]}
    source 5: {sources[4]}
    """

    # call the mistral API to generate a response
    api_key = os.environ["MISTRAL_API_KEY"]
    model = "mistral-large-latest"

    client = MistralClient(api_key=api_key)

    chat_response = client.chat(
        model=model, messages=[ChatMessage(role="user", content=prompt)]
    )

    response = chat_response.choices[0].message.content

    # format the response
    formated_response = response.replace("\n", "")
    return formated_response


if __name__ == "__main__":
    out_f = "./data/llm_responses.csv"
    in_f = "./data/retrieved_sources.csv"  # do not provide an input file if you want to use command line arguments

    # get the total number of arguments
    n = len(sys.argv)

    # if just a query is provided
    if n == 2:
        query = sys.argv[1]
        responses = [(query, None)]  # get sources from poliy-chat
        print("need to call policy chat")

    else:  # a csv file is passed with sources already generated
        print("sources already generated")
        responses = []
        with open(in_f, mode="r", newline="") as f_in:
            reader = csv.reader(f_in)
            for row in reader:
                query = row[0]
                sources = row[1:]
                responses.append((query, sources))

    # open the output file for writing
    with open(out_f, mode="w", newline="") as f_out:
        writer = csv.writer(f_out)
        # write the header
        writer.writerow(["query", "response"])

        # generate the llm response
        for row in responses[1:]: #skips the header
            query = row[0]
            sources = row[1]
            llm_response = get_llm_response(query, sources)

            # format the row to write
            row = [query] + [llm_response] + [""]
            # write the row
            writer.writerow(row)
    print(output)
