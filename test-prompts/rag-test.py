import json
import csv
import requests

# headers for the csv file
headers = ['prompt', 'source1', 'source2', 'source3', 'source4', 'source5']

# save the query and source texts to a csv file
with open("responses.csv", mode='w', newline='') as output_file:
    writer = csv.writer(output_file)
    writer.writerow(headers)
    responses = []

    # read in the queries
    with open('queries.csv', mode='r') as input_file:

        queries = csv.reader(input_file)
        for row in queries:
            if not row:
                continue
            query = row[0].strip()
            result = requests.post("http://localhost:8000/ask", json={"query": query})
            response_data = result.json()

            # get the text from the sources
            sources = response_data.get('source_documents', [])
            source_texts = [json.loads(source['page_content']).get('text', '') for source in sources]

            # write to the csv file
            full_row = [query] + source_texts + ['']
            writer.writerow(full_row)

            responses.append({"query": query, "response": response_data})

        # also save the prompt and response to a json file
        with open("responses.json", "w") as f:
            json.dump(responses, f, indent=4)

print("all done")
