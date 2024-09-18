from flask import Flask, request, jsonify
import os
from flask_cors import CORS
from dataset_utils import Dataset
from rerank_methods import simple_rerank, score_based_rerank

app = Flask(__name__)
CORS(app)
global dataset
dataset = Dataset("dataset/data.tsv")


@app.route("/api/rank", methods=["POST", "GET"])
def rank_docs():
    # Extract data from the request
    data = request.json
    query = data.get("query", "")
    doc_list = data.get("doc_list", [])
    liked_doc_list = data.get("liked_doc_list", [])
    disliked_doc_list = data.get("dislike_doc_list", [])
    liked_others = data.get("liked_others", [])
    disliked_others = data.get("dislike_others", [])
    rerank_method = data.get("rerank_method", "simple")

    if rerank_method == "simple":
        # Implement a simple ranking algorithm
        ranked_doc_list = simple_rerank(doc_list, liked_doc_list, disliked_doc_list)
    else:
        ranked_doc_list = score_based_rerank(
            query, doc_list, liked_doc_list, disliked_doc_list, dataset
        )

    suggested_query = generate_suggested_query(query, liked_others, disliked_others)

    # Prepare the response
    response = {"ranked_doc_list": ranked_doc_list, "suggested_query": suggested_query}

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
