from dotenv import  load_dotenv
import os
import requests

from configs.reranker_config import reranker_config

load_dotenv()

def rerank_documents(query: str, documents: list[str]) -> list[float]:
    headers = {
    "Authorization": f"Bearer {reranker_config.text_rerank_api_key}",
    "Content-Type": "application/json"
    }

    payload = {
        "model": reranker_config.text_rerank_model,
        "query": query,
        "documents": documents,
        "return_documents": False,
        "top_n": len(documents),
        "instruction": reranker_config.text_rerank_instruct
    }

    response = requests.post(
        url=f"{os.getenv("OPENAI_API_BASE")}/rerank",
        headers=headers,
        json=payload,
        timeout=30
    )

    response.raise_for_status()

    response_data = response.json()
    scores = [0.0] * len(documents)
    for item in response_data.get("results"):
        index = item.get("index")
        score = item.get("relevance_score")
        scores[int(index)] = float(score)
    return scores
