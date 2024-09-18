Hackathon Rerank! Right now!

For testing, 
- Run `streamlit run backend_test_ui.py` in one terminal, and `python app.py` in another. 
- Open the link under the streamlit run command, usually `http://localhost:8501`
- Edit the inputs to the API and press `Sent to API` to see the respnse printed. 


To use different rerank methods, use the following syntax:
```py
from rerank_methods import simple_rerank, score_based_rerank, gpt_based_rerank
from dataset_utils import Dataset

dataset = Dataset()
query = "test query 1"
doc_list = dataset.df[dataset.df["Query"] == query]["MUrlKey"].tolist()
liked_doc_list = ["key1", "key9", "key7"]
disliked_doc_list = ["key4", "key6"]
reranked_list = gpt_based_rerank(
    query,
    dataset.df[dataset.df.Query == query]["MUrlKey"].values.tolist(),
    liked_doc_list,
    disliked_doc_list,
    dataset
)
print(reranked_list)
```

Output:
```json
{
    ranked_titles: ['Example Page 1', 'Example Page 9', 'Example Page 7', 'Example Page 5', 'Example Page 8', 'Example Page 10', 'Example Page 11', 'Example Page 12', 'Example Page 13', 'Example Page 4', 'Example Page 6']
}
```