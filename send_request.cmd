curl -X GET https://rerank.onrender.com/api/rank -H "Content-Type: application/json" -d "{ \"query\": \"example search query\", \"doc_list\": [ {\"doc_id\": 1, \"title\": \"Doc 1\", \"murl\": \"http://example.com/1\", \"purl\": \"http://example.com/preview/1\", \"snippet\": \"Snippet for document 1\"}, {\"doc_id\": 2, \"title\": \"Doc 2\", \"murl\": \"http://example.com/2\", \"purl\": \"http://example.com/preview/2\", \"snippet\": \"Snippet for document 2\"}, {\"doc_id\": 3, \"title\": \"Doc 3\", \"murl\": \"http://example.com/3\", \"purl\": \"http://example.com/preview/3\", \"snippet\": \"Snippet for document 3\"} ], \"liked_doc_list\": [1], \"dislike_doc_list\": [2], \"liked_others\": [\"related search example\"], \"dislike_others\": [\"related image example\"] }"
