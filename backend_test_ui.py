import streamlit as st
import requests
import json

# Streamlit app layout
st.title("Flask API Tester")
st.write("Enter a list to be reranked by the Flask API:")

# Input field for the list
input_list = st.text_area(
    "Input Doc List (comma-separated values)",
    """{
      "doc_id": 1,
      "title": "Doc 1",
      "murl": "http://example.com/1",
      "purl": "http://example.com/preview/1",
      "snippet": "Snippet for document 1"
    },
    {
      "doc_id": 2,
      "title": "Doc 2",
      "murl": "http://example.com/2",
      "purl": "http://example.com/preview/2",
      "snippet": "Snippet for document 2"
    },
    {
      "doc_id": 3,
      "title": "Doc 3",
      "murl": "http://example.com/3",
      "purl": "http://example.com/preview/3",
      "snippet": "Snippet for document 3"
    }"""
)
query = st.text_area("Search Query", "test query")
liked_doc_list = st.text_area("Liked List (comma-separated ids)", "1,3")
disliked_doc_list = st.text_area("Disliked List (comma-separated ids)", "2")
liked_others = st.text_area("Liked Others (comma-separated terms)", "other query 1, other query 2")
disliked_others = st.text_area("Disliked Others (comma-separated terms)", "other query 3")

# Button to send the request
if st.button("Send to API"):
    # Convert input string to list
    input_list = json.loads("[" + input_list + "]")
    liked_doc_list = liked_doc_list.split(",")
    disliked_doc_list = disliked_doc_list.split(",")
    liked_others = liked_others.split(",")
    disliked_others = disliked_others.split(",")

    # Prepare the JSON payload
    payload = {
        "query": query,
        "doc_list": input_list,
        "liked_doc_list": liked_doc_list,
        "dislike_doc_list": disliked_doc_list,
        "liked_others": liked_others,
        "dislike_others": disliked_others,
    }

    # Send the request to the Flask API
    response = requests.post("http://localhost:5000/api/rank", json=payload)

    # Display the response
    if response.status_code == 200:
        st.write(response.json())
    else:
        st.write("Error:", response.status_code, response.text)
