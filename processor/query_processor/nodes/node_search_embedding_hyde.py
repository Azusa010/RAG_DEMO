from pymilvus.milvus_client import milvus_client

from configs.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.prompt.search_embedding_hyde import HYDE_PROMPT
from processor.query_processor.state import QueryGraphState
from processor.utils.embedding_utils import generate_embeddings
from processor.utils.llm_utils import get_llm_client
from processor.utils.milvus_utils import create_hybrid_search_requests, hybrid_search, get_milvus_client
from processor.utils.serialize_json import serialize_json
from tool.logger import logger


class NodeSearchEmbeddingHyde(NodeBase):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_search_embedding_hyde"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")

        try:
            hyde_doc = self._step_1_create_hyde_doc(rewritten_query)

            res = self._step_2_search_embedding_hyde(
                rewritten_query=rewritten_query,
                hyde_doc=hyde_doc,
                item_names=item_names
            )

            return {
                "hyde_embedding_chunks": res,
                "hyde_doc": hyde_doc,
            }
        except Exception as e:
            logger.exception(f"假设性文档向量搜索失败: {e}")
            return {}


    def _step_1_create_hyde_doc(self, rewritten_query: str) -> str:
        logger.info(rewritten_query)
        try:
            llm = get_llm_client()
            hyde_prompt = HYDE_PROMPT.format(rewritten_query=rewritten_query)
            hyde_doc = llm.invoke(hyde_prompt).content
            return hyde_doc
        except Exception as e:
            logger.exception(f"步骤1: 生成假设文档失败: {e}")
            raise e

    def _step_2_search_embedding_hyde(
            self,
            rewritten_query: str,
            hyde_doc: str,
            item_names=None
    ):
        try:
            combined_text = rewritten_query + " " + hyde_doc

            embeddings = generate_embeddings([combined_text])
            dense_vector = embeddings.get("dense")[0]
            sparse_vector = embeddings.get("sparse")[0]

            collection_name = milvus_config.chunks_collection
            expr = None
            if item_names:
                expr = f'item_name in {item_names}'
                logger.info(f"步骤2: 过滤条件: {expr}")
            else:
                logger.info("步骤2: 未指定商品名过滤，将全库检索")

            reqs = create_hybrid_search_requests(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                expr=expr,
                limit=10
            )

            logger.info("步骤2: 开始执行 Milvus 混合检索...")
            client = get_milvus_client()
            res = hybrid_search(
                client=client,
                collection_name=collection_name,
                reqs=reqs,
                ranker_weights=(0.8, 0.2),
                output_fields=["chunk_id", "content", "item_name"],
            )

            return res[0] if res else []
        except Exception as e:
            logger.error(f"步骤2: 检索过程发生异常: {e}")
            raise e


if __name__ == "__main__":
    init_state = {
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？",
        "item_names": ["BrotherHAK180烫金机", "BrotherHAK-180烫金机"]
    }
    node_search_embedding_hyde = NodeSearchEmbeddingHyde()
    result = node_search_embedding_hyde(init_state)
    logger.info(serialize_json(result, indent=4))
