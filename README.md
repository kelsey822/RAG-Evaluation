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