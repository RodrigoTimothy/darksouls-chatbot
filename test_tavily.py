import requests

headers = {
    "Authorization": "Bearer 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",
    "Content-Type": "application/json"
}

json_data = {
    "query": "Quem é Gael em Dark Souls?",
    "include_answer": True
}

res = requests.post("https://api.tavily.com/search", headers=headers, json=json_data)
print(res.status_code)
print(res.json())