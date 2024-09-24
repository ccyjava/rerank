import streamlit as st
import requests
import json
from dataset_utils import Dataset

# Initialize session state for liked and disliked lists
st.session_state.liked_doc_list = []
st.session_state.disliked_doc_list = []
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False

global dataset
dataset = Dataset("dataset/DocImageTuringMMFeature_Processed.csv", preprocessed=True)

# Streamlit app layout
st.title("Flask API Tester")
st.write("Enter a list to be reranked by the Flask API:")

# Input field for the list
query_list = dataset.df.Query.unique()
query = st.radio("Select a Search Query", query_list[::-1])
if st.button("Load Docs for query"):
    st.session_state.docs_loaded = True

st.session_state.original_ranked_url_list = []
if st.session_state.docs_loaded:
    doc_list = dataset.get_doc_list(query)
    top_k = st.slider("Number of Docs to Show", 1, len(doc_list), 20)
    st.session_state.original_ranked_url_list = dataset.df[
        dataset.df["Query"] == query
    ]["MUrl"].values.tolist()
    st.write("Original Ranked URLs:")

    liked_doc_list = []
    disliked_doc_list = []
    cols = st.columns(5)  # Adjust the number of columns as needed
    
    for idx, url in enumerate(st.session_state.original_ranked_url_list[:top_k]):
        doc_id = doc_list[idx]
        col = cols[idx % 5]
        col.image(url, use_column_width=True)
        liked_or_disliked = col.radio(f"d{doc_id}", ("None", "Like", "Dislike"))
        if liked_or_disliked == "Like":
            st.session_state.liked_doc_list.append(doc_id)
        elif liked_or_disliked == "Dislike":
            st.session_state.disliked_doc_list.append(doc_id)
    st.write(
        f"Liked Doc ids: {st.session_state.liked_doc_list}, Disliked Doc ids: {st.session_state.disliked_doc_list}"
    )

# liked_doc_list = st.text_area("Liked List (comma-separated ids)", "d559,d560")
# disliked_doc_list = st.text_area("Disliked List (comma-separated ids)", "d561")
# liked_others = st.text_area(
#     "Liked Others (comma-separated terms)", ""
# )
# disliked_others = st.text_area(
#     "Disliked Others (comma-separated terms)", ""
# )
method = st.radio("Rerank Method", ("simple", "score_based"))

# Button to send the request
if st.button("Send to API"):
    liked_doc_list = st.session_state.liked_doc_list
    disliked_doc_list = st.session_state.disliked_doc_list
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

    st.write("Liked Images:")
    liked_url_list = response.json().get("liked_urls", [])
    cols = st.columns(3)  # Adjust the number of columns as needed
    for idx, url in enumerate(liked_url_list):
        col = cols[idx % 3]
        col.markdown(
            f"""
            <div style="background-color: rgba(0, 255, 0, 0.2); padding: 10px;">
                <img src="{url}" style="width: 100%;">
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("Disliked Images:")
    disliked_url_list = response.json().get("disliked_urls", [])
    cols = st.columns(5)  # Adjust the number of columns as needed
    for idx, url in enumerate(disliked_url_list):
        col = cols[idx % 5]
        col.markdown(
            f"""
            <div style="background-color: rgba(255, 0, 0, 0.2); padding: 10px;">
                <img src="{url}" style="width: 100%;">
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Display the images in a matrix
    st.write("Reranked Images:")
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
