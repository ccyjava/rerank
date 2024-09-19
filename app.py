from flask import Flask, request, jsonify
import os
from flask_cors import CORS
from dataset_utils import Dataset
from rerank_methods import simple_rerank, score_based_rerank

app = Flask(__name__)
CORS(app)
global dataset
dataset = Dataset("dataset/DocImageTuringMMFeature.tsv")


@app.route("/api/rank", methods=["POST", "GET"])
def rank_docs():
    # Extract data from the request
    data = request.json
    query = data.get("query", "")
    doc_list = dataset.get_doc_list(query=query)
    liked_doc_list = data.get("liked_doc_list", [])
    disliked_doc_list = data.get("dislike_doc_list", [])
    liked_others = data.get("liked_others", [])
    disliked_others = data.get("dislike_others", [])
    rerank_method = data.get("rerank_method", "simple")

    if rerank_method == "simple":
        # Implement a simple ranking algorithm
        ranked_doc_list = simple_rerank(doc_list, liked_doc_list, disliked_doc_list)
    elif rerank_method == "score_based":
        ranked_doc_list = score_based_rerank(
            query, doc_list, liked_doc_list, disliked_doc_list, dataset
        )
    else:
        print("Invalid rerank method, returning the list as is")
        ranked_doc_list = doc_list

    suggested_query = generate_suggested_query(query, liked_others, disliked_others)
    liked_urls = [
        dataset.df[dataset.df["doc_id"] == doc_id]["MUrl"].iloc[0]
        for doc_id in liked_doc_list
    ]
    disliked_urls = [
        dataset.df[dataset.df["doc_id"] == doc_id]["MUrl"].iloc[0]
        for doc_id in disliked_doc_list
    ]
    # Prepare the response
    response = {
        "original_ranked_doc_list": doc_list,
        "original_ranked_url_list": [
            dataset.df[dataset.df["doc_id"] == doc_id]["MUrl"].iloc[0]
            for doc_id in doc_list
        ],
        "ranked_doc_list": ranked_doc_list,
        "suggested_query": suggested_query,
        "ranked_url_list": [
            dataset.df[dataset.df["doc_id"] == doc_id]["MUrl"].iloc[0]
            for doc_id in ranked_doc_list
        ],
        "liked_urls": liked_urls,
        "disliked_urls": disliked_urls,
    }

    return jsonify(response)


def generate_suggested_query(query, liked_others, dislike_others):
    # Example suggestion logic: modify query based on likes and dislikes
    suggested_query = [query]  # Start with the original query

    # Add suggestions based on liked items
    if liked_others:
        suggested_query.append(
            f"{query} {liked_others[0]}"
        )  # Example of adding a liked term

    # Handle disliked items
    if dislike_others:
        suggested_query = [
            q for q in suggested_query if dislike_others[0] not in q
        ]  # Remove disliked terms

    return suggested_query


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
