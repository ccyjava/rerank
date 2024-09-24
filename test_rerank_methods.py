import unittest
from rerank_methods import simple_rerank, score_based_rerank
import numpy as np


class MockDataset:
    def __init__(self):
        np.random.seed(42)
        self.embeddings = np.random.random((4, 16))

    def get_doc_embedding_batch(self, query, doc_list):
        # Mock embeddings based on doc_list length
        return np.array([self.embeddings[doc["doc_id"]] for doc in doc_list])


class TestRerankMethods(unittest.TestCase):
    def setUp(self):
        self.doc_list = [
            {"doc_id": "doc1"},
            {"doc_id": "doc2"},
            {"doc_id": "doc3"},
            {"doc_id": "doc4"},
        ]
        self.liked_doc_list = ["doc1", "doc3"]
        self.disliked_doc_list = ["doc2"]
        self.dataset = MockDataset()

    def test_simple_rerank(self):
        result = simple_rerank(
            self.doc_list, self.liked_doc_list, self.disliked_doc_list
        )
        expected = ["doc1", "doc3", "doc4"]
        self.assertEqual(result, expected)

    def test_simple_rerank_no_likes_dislikes(self):
        result = simple_rerank(self.doc_list, [], [])
        expected = ["doc1", "doc2", "doc3", "doc4"]
        self.assertEqual(result, expected)

    def test_score_based_rerank(self):
        result = score_based_rerank(
            "query",
            self.doc_list,
            self.liked_doc_list,
            self.disliked_doc_list,
            self.dataset,
        )
        expected = ["doc1", "doc3", "doc4", "doc2"]
        self.assertEqual([doc["doc_id"] for doc in result], expected)

    def test_score_based_rerank_no_likes_dislikes(self):
        result = score_based_rerank("query", self.doc_list, [], [], self.dataset)
        expected = ["doc4", "doc3", "doc2", "doc1"]
        self.assertEqual([doc["doc_id"] for doc in result], expected)


if __name__ == "__main__":
    unittest.main()
