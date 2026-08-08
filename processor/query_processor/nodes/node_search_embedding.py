import json

from configs.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from processor.utils.embedding_utils import generate_embeddings
from processor.utils.milvus_utils import create_hybrid_search_requests, get_milvus_client, hybrid_search
from processor.utils.serialize_json import serialize_json
from tool.logger import logger


class NodeSearchEmbedding(NodeBase):
    """
   节点功能：基于已确认主体名+改写后的用户问题，执行Milvus向量数据库混合检索
   """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_search_embedding"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        try:
            query = state.get("rewritten_query")
            item_names = state.get("item_names")

            collection_name = milvus_config.chunks_collection
            client = get_milvus_client()
            if not client.has_collection(collection_name):
                logger.warning(f"Milvus 集合不存在：{collection_name}")
                return {"embedding_chunks": []}

            stats = client.get_collection_stats(collection_name)
            row_count = int(stats.get("row_count", 0))

            if row_count == 0:
                logger.info(f"Milvus 集合为空：{collection_name}")
                return {"embedding_chunks": []}

            embeddings = generate_embeddings([query])
            dense_vec = embeddings.get("dense")[0]
            sparse_vec = embeddings.get("sparse")[0]

            expr = None
            if item_names:
                expr = f'item_name in {item_names}'
                logger.info(f"过滤条件: {expr}")
            else:
                logger.info("未指定商品名过滤，将全库检索")

            reqs = create_hybrid_search_requests(dense_vector=dense_vec, sparse_vector=sparse_vec, limit=10, expr=expr)
            logger.info("开始执行 Milvus 混合检索...")
            res = hybrid_search(
                client=client,
                collection_name=collection_name,  # 检索的目标集合名（文本片段向量集合）
                reqs=reqs,  # 构造好的混合搜索请求对象（稠密+稀疏）
                ranker_weights=(0.8, 0.2),  # 稠/稀疏向量评分权重配比，各占50%（可按业务调优）
                output_fields=["chunk_id", "content", "item_name"]  # 指定返回的业务字段
            )

            return {"embedding_chunks": res[0] if res else []}
        except Exception as e:
            logger.exception(f"向量搜索失败: {e}")
            return {}


if __name__ == "__main__":
    init_state = {
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？",
        "item_names": ["BrotherHAK180烫金机", "BrotherHAK-180烫金机"]
    }
    node_search_embedding = NodeSearchEmbedding()
    result = node_search_embedding(init_state)
    logger.info(serialize_json(result, indent=4))
