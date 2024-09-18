import requests
AZURE_OPENAI_API_KEY="378e999d50fc4fc4bc3b88b01134cfa0"


def gpt_rerank(search_query, liked_titles, disliked_titles, all_titles):
    # Configuration
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }

    # Payload for the request
    payload = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                    "type": "text",
                    "text": """
                    You are a helpful assistant that can provide a reranking of search results
                    based on a list of liked and disliked document titles, and a search query.
                    You will be provided the input in the following format:
                    {
                        disliked_titles: ["title1", "title2"],
                        liked_titles: ["title3", "title4"],
                        search_query: "original query",
                        all_titles: ["title1", "title2", "title3", "title4", "title5", "title6", "title7", "title8", "title9", "title10"]
                    }
                    Your task is to generate a ranking of all_titles based on the input provided.
                    You may consider the similarity of the liked titles and dissimilarity of the disliked titles, 
                    as well as the original search query.
                    The output must be in the following format:
                    {
                        ranked_titles: ["title3", "title1", "title2", "title4", "title5", "title6", "title7", "title8", "title9", "title10"]
                    }
                    """
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{{ disliked_titles: {disliked_titles}, liked_titles: {liked_titles}, all_titles: {all_titles}, search_query: '{search_query}' }}"
                    }
                ]
            }

        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 1000
    }

    ENDPOINT = "https://ridasgu-aoai.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-02-15-preview"

    # Send request
    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    except requests.RequestException as e:
        raise SystemExit(f"Failed to make the request. Error: {e}")

    return response.json()["choices"][0]["message"]["content"]