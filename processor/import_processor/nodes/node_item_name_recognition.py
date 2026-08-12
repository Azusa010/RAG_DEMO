# processor/import_processor/nodes/node_item_name_recognition.py
import json
import logging
from typing import Tuple, List, Dict

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pymilvus import MilvusClient, DataType
from scipy.optimize import bracket
from torchgen.api.cpp import return_names

from configs.lm_config import lm_config
from configs.milvus_config import milvus_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState
from processor.utils.embedding_utils import generate_embeddings
from processor.utils.milvus_utils import get_milvus_client, escape_milvus_string


class NodeItemNameRecognition(BaseNode):
    """
    主体识别节点：主体识别与标签提取
    """

    name = "node_item_name_recognition"

    def process(self, state: ImportGraphState):
        # 步骤1：提取并校验输入
        file_title, chunks = self._step_1_get_inputs(state)

        # 步骤2：构建大模型识别的上下文
        context = self._step_2_build_context(chunks)

        # 步骤3：调用大模型识别商品名称
        item_name = self._step_3_call_llm(file_title, context)

        # 步骤4：回填商品名称到状态和切片
        self._step_4_update_chunks(state, chunks, item_name)
        #
        # 步骤5：为商品名称生成稠密/稀疏向量
        dense_vector, sparse_vector = self._step_5_generate_vectors(item_name)
        #
        # 步骤6：将数据存入Milvus向量数据库
        self._step_6_save_to_milvus(state, file_title, item_name, dense_vector, sparse_vector)
        #
        # 打印识别结果
        self.logger.info(f"--- 识别完成: {item_name} ---")

        return state

    def _step_1_get_inputs(self, state: ImportGraphState) -> Tuple[str, List[Dict]]:
        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(field_name="file_title", message="文件标题不能为空", expected_type=str)

        chunks = state.get("chunks")
        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

        if not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)

        return file_title, chunks

    def _step_2_build_context(self, chunks: List[Dict]) -> str:
        parts: List[str] = []
        total_chars = 0
        for idx, chunk in enumerate(chunks[:self.config.item_name_chunk_k], start=1):
            chunk_title = chunk.get("title").strip()
            chunk_content = chunk.get("content").strip()

            piece = f"【切片{idx}】\n标题{chunk_title}\n内容：{chunk_content}"
            parts.append(piece)

            total_chars += len(piece)

            if total_chars >= self.config.item_name_chunk_size:
                self.logger.warning(f"累计字符数{total_chars}已超过限制{self.config.item_name_chunk_size}，停止切分")
                break

        context = "\n\n".join(parts).strip()
        final_context = context[:self.config.item_name_chunk_size]
        return final_context

    def _step_3_call_llm(self, file_title: str, context: str) -> str:
        if not context:
            return file_title
        try:
            user_prompt = f"""
请从以下信息中识别出商品名称与型号：
文件名：{file_title}

正文切片（用于辅助识别）：
{context}

要求：
1. 返回内容为字符串形式，最好是带品牌、型号和名称的完整商品名称。比如：苏伯尓5000W大功率电磁炉；
2. 返回结果应该只包含商品名称，不要添加任何解释或其他内容；
3. 如果无法识别商品名称,请返回空字符串。
"""
            model = init_chat_model(
                model=lm_config.llm_model,
                base_url=lm_config.base_url,
                model_provider="openai",
                temperature=lm_config.llm_temperature,
                extra_body={"enable_thinking": False},
            )

            messages = [
                SystemMessage(content="你是一个专业的商品名称识别模型，请根据提供的信息，识别商品名称。"),
                HumanMessage(content=user_prompt),
            ]

            response = model.invoke(messages)
            item_name = response.content

            item_name = (item_name.replace(" ", "")
                         .replace("\n", "")
                         .replace("\t", "")
                         .replace("\r", ""))

            if not item_name:
                return file_title
            return item_name

        except Exception as e:
            self.logger.error(f"大模型调用异常：{e}")
            return file_title

    def _step_4_update_chunks(self, state: ImportGraphState, chunks: List[Dict[str, str]], item_name: str):

        state["item_name"] = item_name

        for chunk in chunks:
            chunk["item_name"] = item_name

        state["chunks"] = chunks

    def _step_5_generate_vectors(self, item_name: str) -> Tuple[List, Dict]:
        if not item_name:
            return None, None

        vectors = generate_embeddings([item_name])
        return vectors["dense"][0], vectors["sparse"][0]

    def _step_6_save_to_milvus(self, state: ImportGraphState, file_title: str, item_name: str, dense_vector,
                               sparse_vector):
        try:
            # 获取milvus客户端
            milvus_client = get_milvus_client()
            if not milvus_client:
                self.logger.warning("无法获取 Milvus 客户端（连接失败），跳过数据保存")
                return

            # 集合初始化
            collection_name = milvus_config.item_name_collection
            if not milvus_client.has_collection(collection_name):
                self._create_item_name_collection(collection_name, milvus_client)

            # 幂等性处理

            safe_item_value = escape_milvus_string(file_title)
            filter_expr = f"item_name=='{safe_item_value}'"

            milvus_client.delete(collection_name=collection_name, filter=filter_expr)

            # 数据插入
            data = {
                "file_title": file_title,
                "item_name": item_name,
            }

            if dense_vector is not None:
                data["dense_vector"] = dense_vector
            if sparse_vector is not None:
                data["sparse_vector"] = sparse_vector
            milvus_client.insert(collection_name=collection_name, data=data)

            state["item_name"] = item_name

        except Exception as e:
            self.logger.warning(f"数据存入Milvus失败，原因：{str(e)}", exc_info=True)

    def _create_item_name_collection(self, collection_name: str, milvus_client: MilvusClient):
        schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)

        schema.add_field(
            field_name="pk",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        )

        schema.add_field(
            field_name="file_title",
            datatype=DataType.VARCHAR,
            max_length=100,
        )

        schema.add_field(
            field_name="item_name",
            datatype=DataType.VARCHAR,
            max_length=100
        )

        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=1024
        )

        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR,

        )

        index_params = milvus_client.prepare_index_params()

        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},

        )

        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",
                "normalize": True,
                "quantization": "none"
            }
        )

        milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )




if __name__ == "__main__":
    setup_logging()
    json_path = r"D:\qdd\hello_RAG\processor\output\Aolynk CB304n Cable网桥 用户手册-5W100-整本手册\chunks.json"
    with open(json_path, 'r') as f:
        json_chunks = json.load(f)

    init_state = {
        "chunks": json_chunks,
        "file_title": "测试文档"
    }

    node = NodeItemNameRecognition()
    result = node(init_state)

    logging.getLogger().info(result)

