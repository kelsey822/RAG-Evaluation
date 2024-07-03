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
python rag-test.py
````
The data will be saved in a csv file containing the query and source texts as separate columns.

# Retrieval metrics
The following metrics are calculated on the retrieval data
  * Precision at k (P@k): Calculates the percentage of top-k sources that were relevant to the query.
  * Mean Reciprocal Rank (MRR): Measures the percentage that the most relevant item was returned at the highest rank.
  * Mean Average Precision (MAP): Measures the average precision at k.
  * Cumulative Gain (CG)
  * Normalized Cumulative Discounted Gain (NCDG)

# Generate the retrieval metrics

Note: responses.csv must be in the same working directory as metric_calc.py to generate metrics

```
# with responses.csv already generated
python metric_calc.py
```
