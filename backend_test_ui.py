import streamlit as st
import requests
import json
from dataset_utils import Dataset

global dataset
dataset = Dataset("dataset/DocImageTuringMMFeature_Processed.csv", preprocessed=True)

# Streamlit app layout
st.title("Flask API Tester")
st.write("Enter a list to be reranked by the Flask API:")

# Input field for the list
query_list = dataset.df.Query.unique()
query = st.radio("Select a Search Query", query_list)
if st.button("Load Docs for query"):
    doc_list = dataset.get_doc_list(query)
    # st.write(f"Docs for query: {doc_list}")
    original_ranked_url_list = dataset.df[dataset.df["Query"] == query]["MUrl"].values.tolist()
    st.write("Original Ranked URLs:")
    if original_ranked_url_list:
      cols = st.columns(5)  # Adjust the number of columns as needed
      for idx, url in enumerate(original_ranked_url_list[:20]):
        col = cols[idx % 5]
        doc_id = doc_list[idx] if idx < len(doc_list) else "Unknown"
        col.image(url, use_column_width=True, caption=f"{doc_id}")
    
liked_doc_list = st.text_area("Liked List (comma-separated ids)", "d559,d560")
disliked_doc_list = st.text_area("Disliked List (comma-separated ids)", "d561")
# liked_others = st.text_area(
#     "Liked Others (comma-separated terms)", ""
# )
# disliked_others = st.text_area(
#     "Disliked Others (comma-separated terms)", ""
# )
method = st.radio("Rerank Method", ("simple", "score_based"))

# Button to send the request
if st.button("Send to API"):
    liked_doc_list = liked_doc_list.strip().split(",")
    disliked_doc_list = disliked_doc_list.strip().split(",")
    # liked_others = [liked_others.strip().split(",")]
    # disliked_others = disliked_others.strip().split(",")
    liked_others = []
    disliked_others = []
    method = method.strip()

    # Prepare the JSON payload
    payload = {
        "query": query,
        "liked_doc_list": liked_doc_list,
        "dislike_doc_list": disliked_doc_list,
        "liked_others": liked_others,
        "dislike_others": disliked_others,
        "rerank_method": method,
    }

    # Send the request to the Flask API
    response = requests.post("http://localhost:5000/api/rank", json=payload)

    # Display the response
    if response.status_code == 200:
        # st.write(response.json())
        pass
    else:
        st.write("Error:", response.status_code, response.text)

    st.write("Liked URLs:")
    liked_url_list = response.json().get("liked_urls", [])
    cols = st.columns(3)  # Adjust the number of columns as needed
    for idx, url in enumerate(liked_url_list):
        col = cols[idx % 3]
        col.image(url, use_column_width=True)

    st.write("Disliked URLs:")
    disliked_url_list = response.json().get("disliked_urls", [])
    cols = st.columns(5)  # Adjust the number of columns as needed
    for idx, url in enumerate(disliked_url_list):
        col = cols[idx % 5]
        col.image(url, use_column_width=True)

    # Display the images in a matrix
    st.write("Ranked URLs:")
    ranked_url_list = response.json().get("ranked_url_list", [])
    if ranked_url_list:
        cols = st.columns(5)  # Adjust the number of columns as needed
        for idx, url in enumerate(ranked_url_list[:20]):
            col = cols[idx % 5]
            col.image(url, use_column_width=True)
    
    st.write("Original Ranked URLs:")
    original_ranked_url_list = response.json().get("original_ranked_url_list", [])
    if original_ranked_url_list:
        cols = st.columns(5)  # Adjust the number of columns as needed
        for idx, url in enumerate(original_ranked_url_list[:20]):
            col = cols[idx % 5]
            col.image(url, use_column_width=True)
