import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class Dataset:
    def __init__(self, file_path="dataset/data.tsv"):
        self.df = pd.read_csv(file_path, sep="\t")
        self.df["TuringMMTextEmbedding"] = self.df["TuringMMTextEmbedding"].apply(eval)

    def get_nearest_neighbors(self, query, murl_key, k=5):
        """
        Get the k nearest neighbors of a given (query, murlkey) pair
        """
        if query not in self.df["Query"].values:
            raise IndexError(f"Query={query} not found in the dataset")
        if murl_key not in self.df[self.df["Query"] == query]["MUrlKey"].values:
            raise IndexError("MUrlKey not found in the dataset for the given query")
        query_embedding = self.df[self.df["Query"] == query][
            self.df["MUrlKey"] == murl_key
        ]["TuringMMTextEmbedding"].values[0]

        other_docs = self.df[self.df["Query"] == query][self.df["MUrlKey"] != murl_key]
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
    result = dataset.get_nearest_neighbors("test query 1", "key1", k=2)
    print(result)
