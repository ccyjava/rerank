import unittest
import pandas as pd
from dataset_utils import Dataset


class TestDataset(unittest.TestCase):
    def setUp(self):
        # Mock dataset
        self.dataset = Dataset()
        data = {
            "Query": ["test query 1", "test query 1", "test query 1", "test query 2"],
            "doc_id": ["d1", "d2", "d3", "d4"],
            "TuringMMTextEmbedding": ["[0.1, 0.2]", "[0.2, 0.3]", "[0.3, 0.4]", "[0.4, 0.5]"]
        }
        self.df = pd.DataFrame(data)
        self.dataset.df = self.df
        self.dataset.df["TuringMMTextEmbedding"] = self.dataset.df["TuringMMTextEmbedding"].apply(eval)

    def test_get_nearest_neighbors_success(self):
        result = self.dataset.get_nearest_neighbors("test query 1", "d1", k=2)
        self.assertEqual(len(result), 2)
        self.assertIn("d2", result["doc_id"].values)
        self.assertIn("d3", result["doc_id"].values)

    def test_get_nearest_neighbors_query_not_found(self):
        with self.assertRaises(IndexError):
            self.dataset.get_nearest_neighbors("nonexistent query", "d1", k=2)

    def test_get_nearest_neighbors_murl_key_not_found(self):
        with self.assertRaises(IndexError):
            self.dataset.get_nearest_neighbors("test query 1", "nonexistent key", k=2)

    def test_get_nearest_neighbors_no_other_docs(self):
        with self.assertRaises(IndexError):
            self.dataset.get_nearest_neighbors("test query 2", "d3", k=2)


if __name__ == "__main__":
    unittest.main()
