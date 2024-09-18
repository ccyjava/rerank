from flask import Flask, request, jsonify
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/rank', methods=['POST'])
def rank_docs():
    # Extract data from the request
    data = request.json
    query = data.get('query', '')
    doc_list = data.get('doc_list', [])
    liked_doc_list = data.get('liked_doc_list', [])
    dislike_doc_list = data.get('dislike_doc_list', [])
    liked_others = data.get('liked_others', [])
    dislike_others = data.get('dislike_others', [])

    # Implement a simple ranking algorithm
    ranked_doc_list = rank_documents(doc_list, liked_doc_list, dislike_doc_list)
    suggested_query = generate_suggested_query(query, liked_others, dislike_others)

    # Prepare the response
    response = {
        'ranked_doc_list': ranked_doc_list,
        'suggested_query': suggested_query
    }

    return jsonify(response)

def rank_documents(doc_list, liked_doc_list, dislike_doc_list):
    # Example ranking logic: prioritize liked docs, exclude disliked docs
    liked_docs = [doc for doc in doc_list if doc['doc_id'] in liked_doc_list]
    neutral_docs = [doc for doc in doc_list if doc['doc_id'] not in liked_doc_list + dislike_doc_list]

    # Combine liked and neutral docs for final ranking
    ranked_docs = liked_docs + neutral_docs
    ranked_doc_ids = [doc['doc_id'] for doc in ranked_docs]

    return ranked_doc_ids

def generate_suggested_query(query, liked_others, dislike_others):
    # Example suggestion logic: modify query based on likes and dislikes
    suggested_query = [query]  # Start with the original query
    
    # Add suggestions based on liked items
    if liked_others:
        suggested_query.append(f"{query} {liked_others[0]}")  # Example of adding a liked term

    # Handle disliked items
    if dislike_others:
        suggested_query = [q for q in suggested_query if dislike_others[0] not in q]  # Remove disliked terms

    return suggested_query

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)