import json
from typing import List, Dict

from bson import ObjectId
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from configs.lm_config import lm_config
from configs.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.prompt.item_name_confirm import ITEM_NAME_EXTRACT_TEMPLATE, \
    ITEM_NAME_EXTRACT_SYSTEM_PROMPT
from processor.query_processor.state import QueryGraphState
from processor.utils.embedding_utils import generate_embeddings
from processor.utils.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from processor.utils.mongo_history_utils import get_recent_messages, save_chat_message, update_message_item_names
from tool.logger import logger


class NodeItemNameConfirm(NodeBase):
    name: str = "node_item_name_confirm"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        session_id, original_query = self._step_1_validate_param(state)
        logger.info(f"步骤1：参数校验通过")

        # 步骤2：获取历史记录
        history = get_recent_messages(session_id)
        logger.info(f"步骤2：获取到 {len(history)} 条历史消息")
        # 更新状态
        state["history"] = history

        # 步骤3：用户初始消息保存
        message_id = save_chat_message(session_id, "user", original_query)
        logger.info(f"步骤3：用户消息已初始保存, ID: {message_id}")

        # 步骤4：提取信息
        extract_res = self._step_4_extract_info(original_query, history)
        item_names = extract_res.get("item_names")
        rewritten_query = extract_res.get("rewritten_query", original_query)
        # 更新状态
        state["rewritten_query"] = rewritten_query
        state["item_names"] = item_names

        # 5. & 6. 如果有提取到商品名，进行搜索和对齐
        align_result = {}
        if len(item_names) > 0:
            query_results = self._step_5_vectorize_and_query(item_names)
            align_result = self._step_6_align_item_names(query_results)
        else:
            logger.info("Node: 未提取到商品名，跳过向量检索")

        # 7. 检查确认状态
        state = self._step_7_check_confirmation(state, align_result, history)

        # 8. 写入最终历史
        self._step_8_write_history(state, session_id, rewritten_query, message_id)
        return state

    def _step_1_validate_param(self, state):
        session_id = state.get("session_id")
        if not session_id:
            raise ValueError("核心参数session_id缺失")

        original_query = state.get("original_query")
        if not original_query:
            raise ValueError("核心参数original_query缺失")

        return session_id, original_query

    def _step_4_extract_info(self, query, history):
        try:
            chat_model = init_chat_model(
                model=lm_config.llm_model,
                model_provider="openai",
                api_key=lm_config.api_key,
                base_url=lm_config.base_url,
                temperature=lm_config.llm_temperature,
                model_kwargs={
                    "response_format": {"type": "json_object"}
                }
            )

            history_text = ""
            for msg in history:
                role = msg.get("role")
                content = msg.get("text")
                history_text += f"{role}: {content}\n"

            user_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(
                history_text=history_text,
                query=query
            )

            messages = [
                SystemMessage(content=ITEM_NAME_EXTRACT_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ]

            response = chat_model.invoke(messages)
            content = response.content

            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")
            result = json.loads(content)

            if "item_names" not in result:
                result["item_names"] = []
            if "rewritten_query" not in result:
                result["rewritten_query"] = query

            result["item_names"] = [
                name.
                replace(" ", "").
                replace("\n", "").
                replace("\t", "").
                replace("\r", "")
                for name in result["item_names"]
            ]

            return result

        except Exception as e:
            logger.error(f"大模型调用异常,{e}")
            return {"item_names": [], "rewritten_query": query}

    def _step_5_vectorize_and_query(self, item_names) -> List[Dict]:
        results = []
        embeddings = generate_embeddings(item_names)

        milvus_client = get_milvus_client()

        if not milvus_client:
            logger.error("连接 Milvus 失败")
            return results

        collection_name = milvus_config.item_name_collection

        for i in range(len(item_names)):
            try:
                dense_vector = embeddings.get("dense")[i]
                sparse_vector = embeddings.get("sparse")[i]

                reqs = create_hybrid_search_requests(dense_vector, sparse_vector, limit=5)
                search_res = hybrid_search(
                    client=milvus_client,
                    collection_name=collection_name,
                    reqs=reqs,
                    ranker_weights=(0.8, 0.2),
                    limit=5,
                    norm_score=True,
                    output_fields=["item_name"]
                )

                matches = []
                if search_res and len(search_res) > 0:
                    for hit in search_res[0]:
                        matches.append({
                            "item_name": hit.get("entity", {}).get("item_name"),
                            "score": hit.get("distance"),
                        })

                results.append({
                    "extracted_name": item_names[i],
                    "matches": matches
                })
            except Exception as e:
                logger.error(f"查询商品名 '{item_names[i]}' 时出错: {e}")
        return results

    def _step_6_align_item_names(self, query_results):
        confirmed_item_names: List[str] = []
        options: List[str] = []

        logger.info(f"步骤6：获得待处理的数据源：{query_results}")

        for res in query_results:
            extracted_name = (res.get("extracted_name", "") or "").strip()
            matches = res.get("matches", []) or []
            if not matches:
                continue

            high = [m for m in matches if m.get("score", 0) > 0.85]
            mid = [m for m in matches if m.get("score", 0) >= 0.6]
            if len(high) > 0:
                for m in high:
                    confirmed_item_names.append(m.get("item_name"))
                continue
            if len(mid) > 0:
                for m in mid[:3]:
                    options.append(m.get("item_name"))
        return {
            "confirmed_item_names": list(set(confirmed_item_names)),
            "options": list(set(options))
        }

    def _step_7_check_confirmation(self, state, align_result, history):
        confirmed = align_result.get("confirmed_item_names",[])
        options = align_result.get("options",[])

        # 分支A：有确认的商品名（高置信度，无需用户确认）
        if confirmed:
            ids_to_update = []
            for msg in history:
                if not msg.get("item_name"):
                    mid = msg.get("_id")
                    if mid:
                        ids_to_update.append(mid)
            if ids_to_update:
                update_message_item_names(ids_to_update,confirmed)
            state["item_names"] = confirmed
            state["answer"] = ""
            return state

        # 分支B：无确认商品名，但有候选商品名（中置信度，需用户明确）
        if options:
            options_str = ",".join(options)
            answer = f"您是想问以下哪个产品：{options_str}？请明确一下型号。"
            state["answer"] = answer
            state["item_names"] = []
            return state

        # 分支C：无确认商品名，且无候选商品名（无匹配结果，需用户重新提供）
        state["answer"] = "抱歉，未找到相关产品，请提供准确型号以便我为您查询。"
        state["item_names"] = []
        return state

    def _step_8_write_history(self, state, session_id, rewritten_query, message_id):
        if state.get("answer"):
            save_chat_message(
                session_id=session_id,  # 会话ID，关联所属会话
                role="assistant",  # 消息角色：助手
                text=state["answer"],  # 消息内容：向用户确认的提示语/无结果提示语
                rewritten_query="",  # 助手消息无需改写查询，设为空
                item_names=state.get("item_names", [])  # 关联的商品名列表（分支B/C均为空）
            )

            # 强制更新本次用户原始问题的关联信息（核心：补充改写查询、商品名）
        save_chat_message(
            session_id=session_id,  # 会话ID，关联所属会话
            role="user",  # 消息角色：用户
            text=state["original_query"],  # 消息内容：用户原始查询
            rewritten_query=rewritten_query,  # 补充step3改写后的完整问题
            item_names=state.get("item_names", []),  # 补充关联的商品名列表
            message_id=message_id  # 消息ID，指定更新已存在的用户消息（而非新增）
        )

        # 返回最终会话状态，供下游节点使用
        return state

if __name__ == "__main__":

    node = NodeItemNameConfirm()
    init_state = {
        "session_id": "test_session_002",
        "original_query": "怎么调他的转印温度？"
    }

    result = node(init_state)

    for items in result["history"]:
        items["_id"] = str(items["_id"])
    json_result = json.dumps(result, ensure_ascii=False, indent=4)
    logger.info(json_result)
