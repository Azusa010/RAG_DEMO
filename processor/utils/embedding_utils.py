from typing import List

from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from configs.embedding_config import embedding_config
from processor.import_processor.base import setup_logging

setup_logging()

_bge_m3_ef = None


def get_bge_m3_ef():
    global _bge_m3_ef

    if _bge_m3_ef is not None:
        return _bge_m3_ef

    model_name = embedding_config.bge_m3_path
    device = embedding_config.bge_device
    use_fp16 = embedding_config.bge_fp16

    _bge_m3_ef = BGEM3EmbeddingFunction(model_name=model_name, device=device, use_fp16=use_fp16)

    return _bge_m3_ef


def generate_embeddings(text: List[str]):
    model = get_bge_m3_ef()
    embeddings = model.encode_documents(text)

    processed_sparse = []
    for i in range(len(text)):
        sparse_indices = embeddings["sparse"].indices[
            embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]].tolist()
        sparse_data = embeddings["sparse"].data[
            embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]].tolist()
        sparse_dict = {k:v for k, v in zip(sparse_indices, sparse_data)}
        processed_sparse.append(sparse_dict)



    return {
        "dense": [emb.tolist() for emb in embeddings["dense"]],
        "sparse": processed_sparse,
    }


if __name__ == "__main__":
    generate_embeddings(['hello world'])
