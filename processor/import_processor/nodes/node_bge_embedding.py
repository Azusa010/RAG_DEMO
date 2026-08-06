# processor/import_processor/nodes/node_bge_embedding.py
import json
import logging
from typing import Dict, List

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState
from processor.utils.embedding_utils import generate_embeddings


class NodeBGEEmbedding(BaseNode):
    """
    混合向量化节点：使用 BGE-M3 模型将文本转换为向量
    """

    name = "node_bge_embedding"

    def process(self, state: ImportGraphState):
        chunks = self._step_1_validate_input(state)
        output_data = self._step_2_generate_embeddings(chunks)


        state["chunks"] = output_data
        return state

    def _step_1_validate_input(self, state: ImportGraphState) -> List[Dict]:
        chunks = state.get("chunks")

        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)
        if not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)
        return chunks

    def _step_2_generate_embeddings(self, chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:

        output_data = []

        batch_size = 5
        for i in range(0,len(chunks),batch_size):
            batch_texts = chunks[i:i+batch_size]
            input_texts = []
            for doc in batch_texts:
                file_title = doc["file_title"]
                content = doc["content"]
                input_texts.append(f"{file_title}\n{content}" if file_title else content)

            docs_embeddings = generate_embeddings(input_texts)
            for j,doc in enumerate(batch_texts):
                item = doc.copy()
                item["dense_vector"] = docs_embeddings['dense'][j]
                item["sparse_vector"] = docs_embeddings['sparse'][j]
                output_data.append(item)
            self.logger.info(f"成功获取第 {i + 1}-{min(i + len(batch_texts), len(chunks))} 项的嵌入。")
        return output_data

if __name__ == "__main__":

    setup_logging()

    json_path = r"D:\qdd\hello_RAG\processor\output\Aolynk CB304n Cable网桥 用户手册-5W100-整本手册\state.json"
    with open(json_path, "r", encoding="utf-8") as f:
        state_json = f.read()

    state = json.loads(state_json)

    init_state = {
        "chunks": state.get("chunks")
    }

    # 执行核心处理流程
    node_bge_embedding = NodeBGEEmbedding()
    result = node_bge_embedding(init_state)

    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))