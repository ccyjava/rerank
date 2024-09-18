from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from gpt_prompting import gpt_rerank


def simple_rerank(doc_list, liked_doc_list, disliked_doc_list):
    # Example ranking logic: prioritize liked docs, exclude disliked docs
    liked_docs = [doc for doc in doc_list if doc["doc_id"] in liked_doc_list]
    neutral_docs = [doc for doc in doc_list if doc["doc_id"] not in liked_doc_list + disliked_doc_list]
    return [doc["doc_id"] for doc in liked_docs + neutral_docs]


def score_based_rerank(query, doc_list, liked_doc_list, disliked_doc_list, dataset):
    # Need to boost the ranking of documents that are similar to liked documents,
    embeddings_liked_docs = dataset.get_doc_embedding_batch(query, liked_doc_list)
    embeddings_disliked_docs = dataset.get_doc_embedding_batch(query, disliked_doc_list) 
    # Need to penalize the ranking of documents that are similar to disliked documents
    all_embeddings = dataset.get_doc_embedding_batch(query, doc_list)
    
    if liked_doc_list:
        # Calculate similarity between all_embeddings and embeddings_liked_docs, and take the highest distance
        liked_docs_similarities = cosine_similarity(all_embeddings, embeddings_liked_docs)
        best_liked_doc_similarity = np.max(liked_docs_similarities, axis=1)
    else:
        best_liked_doc_similarity = np.zeros(len(doc_list))
    if disliked_doc_list:
        # Calculate similarity between all_embeddings and embeddings_disliked_docs
        dislked_docs_similarities = cosine_similarity(all_embeddings, embeddings_disliked_docs)
        best_disliked_doc_similarity = np.max(dislked_docs_similarities, axis=1)
    else:
        best_disliked_doc_similarity = np.zeros(len(doc_list))

    # Calculate the score for each document
    all_docs_score = best_liked_doc_similarity - best_disliked_doc_similarity
    ranked_doc_indices = np.argsort(all_docs_score)[::-1]
    return [doc_list[i] for i in ranked_doc_indices]


def gpt_test_based_rerank(query, doc_list, liked_doc_list, disliked_doc_list, dataset):
    # get titles of liked and disliked docs
    liked_titles = [
        dataset.df[(dataset.df["Query"] == query) & (dataset.df["MUrlKey"] == doc_id)]["PageTitle"].iloc[0]
        for doc_id in liked_doc_list
    ]
    disliked_titles = [
        dataset.df[(dataset.df["Query"] == query) & (dataset.df["MUrlKey"] == doc_id)]["PageTitle"].iloc[0]
        for doc_id in disliked_doc_list
    ]
    all_titles = [
        dataset.df[(dataset.df["Query"] == query) & (dataset.df["MUrlKey"] == doc_id)]["PageTitle"].iloc[0]
        for doc_id in doc_list
    ]
    # print(liked_titles, disliked_titles, all_titles)
    # generate a new query based on liked and disliked titles
    # query gpt model with the new query
    return gpt_rerank(query, liked_titles, disliked_titles, all_titles)