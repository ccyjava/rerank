import requests
from dotenv import dotenv_values

config = dotenv_values(".env")

def gpt_rerank(search_query, liked_titles, disliked_titles, all_titles):
    # Configuration
    headers = {
        "Content-Type": "application/json",
        "api-key": config["API_KEY"],
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
                    You may consider the similarity of each title to that of
                    the liked titles and the dissimilarity to the disliked titles to make your ranking
                    decisions, as well as the original search query for relevance.
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

    endpoint = config["ENDPOINT"]
    # Send request
    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    except requests.RequestException as e:
        raise SystemExit(f"Failed to make the request. Error: {e}")

    return response.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    # Example usage
    search_query = "best programming language"
    liked_titles = ["Python", "JavaScript"]
    disliked_titles = ["Java"]
    all_titles = ["Python", "JavaScript", "Java", "C++", "Ruby", "Go", "Swift", "Kotlin", "Rust", "TypeScript"]

    ranked_titles = gpt_rerank(search_query, liked_titles, disliked_titles, all_titles)
    print(ranked_titles)
