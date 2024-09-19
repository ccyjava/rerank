from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dataset_utils import Dataset

# from gpt_prompting import gpt_rerank


def simple_rerank(doc_list, liked_doc_list, disliked_doc_list):
    # Example ranking logic: prioritize liked docs, exclude disliked docs
    neutral_docs = [doc for doc in liked_doc_list + disliked_doc_list]
    return [doc for doc in liked_doc_list + neutral_docs]


def score_based_rerank(
    query,
    doc_list,
    liked_doc_list,
    disliked_doc_list,
    dataset: Dataset,
    original_order_weight=2.0,
    liked_doc_weight=1.0,
    disliked_doc_weight=10.0,
):
    # Need to boost the ranking of documents that are similar to liked documents,
    # and penalize the ranking of documents that are similar to disliked documents
    embeddings_liked_docs = dataset.get_doc_embedding_batch(
        query=query, doc_ids=liked_doc_list
    )
    embeddings_disliked_docs = dataset.get_doc_embedding_batch(
        query=query, doc_ids=disliked_doc_list
    )
    # Need to penalize the ranking of documents that are similar to disliked documents
    all_embeddings = dataset.get_doc_embedding_batch(query=query, doc_ids=doc_list)
    print(
        "Debugging: ",
        all_embeddings.shape,
        embeddings_liked_docs.shape,
        embeddings_disliked_docs.shape,
    )

    if liked_doc_list:
        # Calculate similarity between all_embeddings and embeddings_liked_docs, and take the highest distance
        liked_docs_similarities = cosine_similarity(
            all_embeddings, embeddings_liked_docs
        )
        best_liked_doc_similarity = np.max(liked_docs_similarities, axis=1)
    else:
        best_liked_doc_similarity = np.zeros(len(doc_list))
    if disliked_doc_list:
        # Calculate similarity between all_embeddings and embeddings_disliked_docs
        dislked_docs_similarities = cosine_similarity(
            all_embeddings, embeddings_disliked_docs
        )
        best_disliked_doc_similarity = np.max(dislked_docs_similarities, axis=1)
    else:
        best_disliked_doc_similarity = np.zeros(len(doc_list))

    # Calculate the score for each document
    ranks = np.arange(len(doc_list))
    rank_scores = np.exp(-ranks)
    all_docs_score = (
        original_order_weight * rank_scores
        + liked_doc_weight * best_liked_doc_similarity
        - disliked_doc_weight * best_disliked_doc_similarity
    )
    for ldoc in liked_doc_list:
        all_docs_score[doc_list.index(ldoc)] += 1e3
    for ddoc in disliked_doc_list:
        all_docs_score[doc_list.index(ddoc)] -= 1

    ranked_doc_indices = np.argsort(-all_docs_score)
    return [doc_list[i] for i in ranked_doc_indices]


def gpt_test_based_rerank(query, doc_list, liked_doc_list, disliked_doc_list, dataset):
    # get titles of liked and disliked docs
    liked_titles = [
        dataset.df[(dataset.df["Query"] == query) & (dataset.df["MUrlKey"] == doc_id)][
            "PageTitle"
        ].iloc[0]
        for doc_id in liked_doc_list
    ]
    disliked_titles = [
        dataset.df[(dataset.df["Query"] == query) & (dataset.df["MUrlKey"] == doc_id)][
            "PageTitle"
        ].iloc[0]
        for doc_id in disliked_doc_list
    ]
    all_titles = [
        dataset.df[(dataset.df["Query"] == query) & (dataset.df["MUrlKey"] == doc_id)][
            "PageTitle"
        ].iloc[0]
        for doc_id in doc_list
    ]
    # print(liked_titles, disliked_titles, all_titles)
    # generate a new query based on liked and disliked titles
    # query gpt model with the new query
    return gpt_rerank(query, liked_titles, disliked_titles, all_titles)
