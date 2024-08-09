# RAG-evaluation
Repository that contains evaluation pipeline for LLMs augmented by Retrieval Augmented Generation.

# Code standards

The following code standards apply to this repository:

1. Formatter: `black`
2. Sorting imports: `isort`
3. Linting: `pylint` with Google `.pylintrc` as starter config
4. Docstrings: Numpydoc

# Dependency management

To set up virtual environment:

```
pyenv local 3.12 # or your method of choice for managing python versions
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To deactivate virtual environment:

```
deactivate
```

To update dependencies:

```
# with virtual environment activated
pip install pip-tools
# add library or libraries to requirements.in
pip-compile requirements.in
pip install -r requirements.txt

```
# Generate the data set

Note: [policy chat][https://github.com/healthmap/policy_chat_backend] needs to be running before generation of the data set
````
# with policy chat running
python retrieve_documents.py
````
The data will be saved in a ``.csv`` file containing the query and retrieved source texts as separate columns.

# Retrieval metrics
The following metrics are calculated on the retrieval data
  * Precision at k (P@k): Calculates the percentage of top-k sources that were relevant to the query.
  * Mean Reciprocal Rank (MRR): Measures the percentage that the most relevant item was returned at the highest rank.
  * Mean Average Precision (MAP): Measures the average precision at k.
  * Cumulative Gain (CG): Measures the total sum of relevances.
  * Normalized Cumulative Discounted Gain (NCDG): Measures the normalized sum of relevances while adding a discount factor for order.


# Generate the retrieval metrics
```
# with responses.csv already generated including human scored relevances
python calculate_retrieval_metrics.py
```

# Generate a response from an LLM
Prior to generating responses from an LLM a MistralAI API key is required set as the environment variable ```MISTRAL_API_KEY```.
There are two ways to generate a response.
To generate a response with no retrieved sources from policy-chat created:

```
python generate_llm_responses.py "Query"
```

Note: An input ``.csv`` file cannot be provided in the script
This will pass the query to policy-chat, generate the relevant sources, and then pass to the LLM.

To generate a response with a ``.csv`` file of queries and pre-generated sources

```
python generate_llm_responses.py
```

Note: An input ``.csv`` file (containing queries and sources) must be provided in the script and in the same directory as the script.

For both methods, the query and LLM response will be saved to a ``.csv`` file after running.


# Generate Metric Plots
With the relevant data frames created, plots can be generated with:

```
python generate_plots.py
```

Used scripts to create the data frames are included in the rag-evaluation directory. Similarly the data for the sample queries is in the data directory. 


# Testing
To test the retrieval metric calculation:

```
python -m pytest
```
