import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class Dataset:
    def __init__(self, file_path="dataset/data.tsv"):
        self.df = pd.read_csv(file_path, sep="\t")
        self.df["doc_id"] = ["d" + str(x) for x in self.df.index.to_list()]
        self.df["TuringMMTextEmbedding"] = self.df["TuringMMv2ImageDecoded"].apply(
            lambda x: [float(i) for i in x.split(" ")] if type(x) == str else x
        )
        self.df = self.df.dropna() # drop rows with nan in embeddings

    def __str__(self):
        return str(self.df.head())

    def get_doc_embedding(self, query, doc_id):
        return self.df[(self.df["Query"] == query) & (self.df["doc_id"] == doc_id)][
            "TuringMMTextEmbedding"
        ].values[0]

    def get_doc_embedding_batch(self, doc_ids):
        return np.array(self.df[self.df["doc_id"].isin(doc_ids)][
            "TuringMMTextEmbedding"
        ].values.tolist())

    def get_nearest_neighbors(self, query, doc_id, k=5):
        """
        Get the k nearest neighbors of a given (query, doc_id) pair
        """
        if query not in self.df["Query"].values:
            raise IndexError(f"Query={query} not found in the dataset")
        if doc_id not in self.df[self.df["Query"] == query]["doc_id"].values:
            raise IndexError("doc_id not found in the dataset for the given query")
        query_embedding = self.get_doc_embedding(query, doc_id)

        other_docs = self.df[self.df["Query"] == query][self.df["doc_id"] != doc_id]
        if other_docs.empty:
            raise IndexError("No other documents found for the query")
        cosine_similarities = cosine_similarity(
            [query_embedding], other_docs["TuringMMTextEmbedding"].tolist()
        )
        # Get the index of the most similar document
        # Add cosine similarity as a column to the other_docs dataframe
        other_docs = other_docs.copy()
        other_docs["cosine_similarity"] = cosine_similarities[0]

        # Get the index of the most similar document
        most_similar_doc_indices = cosine_similarities.argsort()[0][::-1][:k]
        most_similar_docs = other_docs.iloc[most_similar_doc_indices]
        return most_similar_docs


if __name__ == "__main__":
    dataset = Dataset()
    print(dataset.df.head())
    result = dataset.get_nearest_neighbors("test query 1", "d1", k=2)
    print(result)
