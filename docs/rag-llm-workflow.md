# Summary
Multiple different methods exist but most commonly, after the query is sent and possibly rewritten or split up into multiple sub-questions, it is sent to the retriever. From there 2 common retrieval methods can be used. For policy-chat, a sparse retriever using the BM25 algorithm is used to return the top-k relevant documents. To further filter out irrelevant information the top-k documents are reranked prior to being sent to the LLM to increase the max likelihood of getting relevant information. The (edited) prompt is then sent to the LLM along with relevant chunks to generate a response. From there the prompt and response can be saved to a json file for easy comparison. Moving forward, it could be interesting to try out different LLMs and consider the dense retrieval method if there is time.

# Notes 
* prompt --> retrieval method
  * the query is transformed into vector representation using the same model that encodes the data
  * different types of query optimization
    * query expansion: one single query is expanded into multiple thought out questions
    * sub-query: relevant sub-questions are generated 
    * query rewrite: a LLM is used to rewrite the questions
  * different retrieval methods
    * iterative retrieval: initial query and generated text is repeatedly searched, could add in irrelevant info
    * recursive retrieval: used to improve depth and relevance of the search, queries are refined iteratively, data is retrieved in order of relevance 
    * adaptive retrieval: LLM are used to judge the answers
  * naive RAG: query is passed as an embedded query and top-k relevant chunks are fetched 
  * after the first retrieval the documents are reranked
  
* retrieval --> llm: reranked chunks are sent to the llm to generate a response along with the original query.
  * sparse retriever: uses algorithms like BM25 for search and match in documents
  * dense retriever: uses approximate nearest neighbor to search within the embedded vectors 
* overall the structure is input -> retriever -> generator -> output
* for easy comparison you can format the query and response into columns side by side in a json file.

