import unittest
import pandas as pd
from dataset_utils import Dataset


class TestDataset(unittest.TestCase):
    def setUp(self):
        # Mock dataset
        data = {
            "Query": ["test query 1", "test query 1", "test query 1", "test query 2"],
            "MUrlKey": ["key1", "key2", "key3", "key4"],
            "TuringMMTextEmbedding": ["[0.1, 0.2]", "[0.2, 0.3]", "[0.3, 0.4]", "[0.4, 0.5]"]
        }
        self.df = pd.DataFrame(data)
        self.dataset = Dataset()
        self.dataset.df = self.df
        self.dataset.df["TuringMMTextEmbedding"] = self.dataset.df["TuringMMTextEmbedding"].apply(eval)

    def test_get_nearest_neighbors_success(self):
        result = self.dataset.get_nearest_neighbors("test query 1", "key1", k=2)
        self.assertEqual(len(result), 2)
        self.assertIn("key2", result["MUrlKey"].values)
        self.assertIn("key3", result["MUrlKey"].values)

    def test_get_nearest_neighbors_query_not_found(self):
        with self.assertRaises(IndexError):
            self.dataset.get_nearest_neighbors("nonexistent query", "key1", k=2)

    def test_get_nearest_neighbors_murl_key_not_found(self):
        with self.assertRaises(IndexError):
            self.dataset.get_nearest_neighbors("test query 1", "nonexistent key", k=2)

    def test_get_nearest_neighbors_no_other_docs(self):
        with self.assertRaises(IndexError):
            self.dataset.get_nearest_neighbors("test query 2", "key4", k=2)

if __name__ == "__main__":
    unittest.main()