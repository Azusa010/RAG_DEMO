import json

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger

class NodeItemNameConfirm(NodeBase):
    name:str = "node_item_name_confirm"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        logger.info(f"【{self.name}】节点逻辑")
        return state


if __name__ == "__main__":

    node = NodeItemNameConfirm()
    init_state = {
        "original_query": "怎么调他的转印温度？"
    }

    result = node(init_state)
    json_state = json.dumps(result, indent=4)
    logger.info(json_state)

