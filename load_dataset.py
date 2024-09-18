import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

file_path = "dataset/data.csv"
df = pd.read_csv(file_path, sep="\t")


def get_nearest_neighbors(query, murl_key):
    # Get the embedding of the given query, murlkey pair
    query_embedding = df[df["Query"] == query][df["MUrlKey"] == murl_key][
        "TuringMMTextEmbedding"
    ].values[0]
    # Get the embeddings of all the rows in the dataset for the same query
    other_docs = df[df["Query"] == query][df["MUrlKey"] != murl_key]
    cosine_similarities = cosine_similarity(
        [query_embedding], other_docs["TuringMMTextEmbedding"].tolist()
    )
    # Get the index of the most similar document
    most_similar_doc_index = cosine_similarities.argmax()
    # Get the most similar document
    most_similar_doc = other_docs.iloc[most_similar_doc_index]
    return most_similar_doc
