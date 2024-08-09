# Week 2 Summary
* Researched different methods of RAG evaluation 
* Decided on a few specific RAG evaluation methods to focus on 
* Set up policy-chat-backend to run locally
* Tested a series of queries for policy chat and saved them to a json file
* Skimmed policy pdfs to create a list of queries

# Resources Reviewed
* [Evaluation Metrics For Information Retrieval][https://amitness.com/posts/information-retrieval-evaluation#problem-setup-2-graded-relevance] 
* [Vector Search][https://github.com/esteininger/vector-search]
* [What is Retrieval Augmented Generation (RAG)?][https://pureinsights.com/blog/2023/what-is-retrieval-augmented-generation-rag/]
* [Evaluation Measures in Information Retrieval][https://www.pinecone.io/learn/offline-evaluation/]
* [Normalized Discounted Cumulative Gain (NDCG) explained][https://www.evidentlyai.com/ranking-metrics/ndcg-metric]
* [RAG and RAU: A Survey on Retrieval-Augmented Language Model in Natural Language Processing][https://arxiv.org/pdf/2404.19543]

# Notes
* RAG evaluation methods
  * multiple varying methods - some take order of the rankings into account, some are only able to evaluate binary relevance 
  * normalized discounted cumulative gain (NDCG)
    * takes the order of the items into account, can handle non binary relevance, normalized data
    * cumulative gain: sum of the relevance of individual items in the list 
    * the discount is how the order is evaluated, usually a log based score keeping method 
    * need to have...
      * ideal ranking to compare the generated relevant chunks to -a ground truth
      * decide on a k parameter (how many items on the list to analyze)
      * decide on a log formula to use for the discount, one method makes a bigger deal of the order than the other
  * mean average precision (MAP) 
    * only applies to binary data
    * calculated as the average of precision@k values for the total return 



