"""Generates a response from a LLM based on the retrieved documents provided by
policy-chat API and the query.The responses are then saved to a csv file with the
query, retrieved sources, and llm generated response.
"""

import csv
import json
import requests
import sys
import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

def pass_to_policy_chat(query: str):
    """Return a list of the retrieved sources from poliy-chat given a query as an input.
    """
    # send POST request to policy-chat API
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
    return source_texts


def get_llm_response(query: str):
    """Takes a command line string query as the argument. Calls policy-chat API to get relevant
    sources and then passes to a LLM for repsonse generation.
    """

    # get the retrieved sources from policy-chat API
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
    model=model,
    messages=[ChatMessage(role="user", content=prompt)]
    )

    response = (chat_response.choices[0].message.content)

    #format the response
    formated_response = response.replace("\n", "")
    print(response)
    return formated_response


if __name__ == "__main__":
    output = "llm_responses.csv"

    #get the total number of arguments
    n = len(sys.argv)

    # open the output file for writing
    with open(output, mode='w', newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["query", "response"])

        #get responses for each of the queries
        for i in range(1, n): #skip the first arg
            query = sys.argv[i]
            response = get_llm_response(query)

            # format the row to write
            row = [query] + [response] + [""]
            #write the row
            writer.writerow(row)
    print(output)
