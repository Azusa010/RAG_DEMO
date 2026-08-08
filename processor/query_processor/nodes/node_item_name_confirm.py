import json
from typing import List, Dict

from bson import ObjectId
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from configs.lm_config import lm_config
from processor.query_processor.base import NodeBase
from processor.query_processor.prompt.item_name_confirm import ITEM_NAME_EXTRACT_TEMPLATE, \
    ITEM_NAME_EXTRACT_SYSTEM_PROMPT
from processor.query_processor.state import QueryGraphState
from processor.utils.embedding_utils import generate_embeddings
from processor.utils.mongo_history_utils import get_recent_messages, save_chat_message
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
                api_key=lm_config.api_key,
                baseurl=lm_config.baseurl,
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

        generate_embeddings()

    def _step_6_align_item_names(self, query_results):
        pass


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
