# processor/import_processor/nodes/node_document_split.py
import json
import logging
import re
from typing import Tuple, List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState

DEFAULT_MAX_CONTENT_LENGTH = 2000
MIN_CONTENT_LENGTH = 500

class NodeDocumentSplit(BaseNode):
    """
    文档切分节点：智能文档切片
    """

    name = "node_document_split"

    def process(self, state: ImportGraphState):
        """
        节点：文档切分（node_document_split）
        整体流程：加载输入→按MD标题初切→长切短合→统计输出→结果备份
        核心目的：将长MD文档切分为长度适中的Chunk，适配大模型上下文窗口和向量检索
        后续扩展点：可在各步骤间新增Chunk元信息补充、自定义切分规则、向量入库前置处理等

        必要参数：task_id、md_path(完整流程中非必要，备份测试用的json文件)、md_content、file_title
        更新参数：chunks

        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # ===================================== 步骤1：加载并标准化输入数据 =====================================
        # 作用：从状态字典提取MD内容/文件标题，统一换行符消除系统差异
        # 输出：标准化后的md_content、文件标题；
        content, file_title = self._step_1_get_inputs(state)

        # ===================================== 步骤2：按MD标题进行初次切分 ===============================
        # 作用：基于Markdown标题（#/##/###）切分文档为独立章节，自动跳过代码块内的伪标题，保证章节语义完整
        # 输出：初切后的章节列表、识别到的有效标题数量、MD原始文本总行数（为后续统计/日志使用）
        sections, title_count, lines_count = self._step_2_split_by_titles(content, file_title)

        # ===================================== 步骤3：无标题场景兜底处理 ===================================
        # 作用：解决MD文档无任何标题的边界情况，避免后续切分逻辑异常
        # 输出：有标题则返回步骤2的章节列表；无标题则将全文封装为单个「无标题」章节，保证数据格式统一
        sections = self._step_3_handle_no_title(content, sections, title_count, file_title)

        return state

    def _step_1_get_inputs(self, state: ImportGraphState) -> Tuple[str, str]:
        """
              【步骤1】获取并预处理输入数据
              功能：从状态字典中提取MD内容/文件标题/最大长度，做基础标准化
              :param state: 项目状态字典（ImportGraphState），包含md_content等核心键
              :return: 标准化后的MD内容/文件标题（无内容则返回None,None）
        """
        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(field_name="file_title", message="文件标题不能为空", expected_type=str)

        md_path = state.get("md_path")
        if not md_path:
            raise StateFieldError(field_name="md_path", message="MD路径不能为空", expected_type=str)

        md_content = state.get("md_content")
        if not md_content:
            raise StateFieldError(field_name="md_content", message="文件内容不能为空", expected_type=str)

        md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
        return md_content, file_title

    def _step_2_split_by_titles(self, content: str, file_title: str) -> Tuple[List[Dict[str, str]], int, int]:
        """
        【步骤2】按Markdown标题初次切分（核心：按#分级切分，跳过代码块内标题）
        LangChain前置预处理：将整份MD按标题拆分为独立章节，为后续精细化切分做基础
        :param content: 标准化后的MD完整内容（字符串）
        :param file_title: 所属文件标题，用于标记章节归属
        :return: 切分后的章节列表/有效标题数量/原始文本总行数
        """
        title_pattern = r'\s*#{1,6}\s+.+'

        lines = content.split("\n")
        sections = []
        title_count = 0
        current_title = ''
        current_lines = []
        in_code_block = False

        def _flush_section():
            if not current_lines:
                return
            sections.append(
                {
                    "title": current_title,
                    "content": '\n'.join(current_lines),
                    "file_title": file_title
                }
            )

        for line in lines:
            stripped_line = line.strip()
            code_block_marker_match = re.match(r'^(`{3,}|~{3,})$', stripped_line)
            if code_block_marker_match:
                marker  = code_block_marker_match.group(1)
                if not in_code_block:
                    in_code_block = True
                    code_block_start_marker  = marker
                elif in_code_block and stripped_line == code_block_start_marker:
                    in_code_block = False
                    code_block_start_marker = None
                current_lines.append(line)
                continue

            is_valid_title = (not in_code_block) and re.match(title_pattern, line)
            if is_valid_title:
                _flush_section()
                current_title = stripped_line
                current_lines = [current_title]
                title_count += 1
                self.logger.info(f"识别标题：{current_title}")
            else:
                current_lines.append(line)
        _flush_section()
        self.logger.info(f"文档粗切（按标题切分）完成，共{len(sections)}个章节，标题数量是{title_count}，文本共有{len(lines)}行")
        return sections, title_count, len(lines)

    def _step_3_handle_no_title(self,content:str,sections:List[Dict[str, str]],title_count:int,file_title:str) -> List[Dict[str, str]]:
        """
        【步骤3】无标题兜底处理
        功能：若MD中未识别到任何标题，将全文作为一个整体处理，避免后续逻辑异常
        :param content: 标准化后的MD完整内容
        :param sections: 步骤2切分后的章节列表
        :param title_count: 步骤2识别的有效标题数量
        :param file_title: 所属文件标题
        :return: 兜底后的章节列表
        """
        if title_count == 0:
            self.logger.warning(f"步骤3：未识别到任何MD标题，将全文作为单个章节处理，文件：{file_title}")
            return [{"title":"无标题","content":content,"file_title":file_title}]
        self.logger.debug(f"步骤3：检测到{title_count}个有效标题，无需兜底处理")
        return sections

    def _step_4_refine_chunks(self,sections: List[Dict[str, str]]) -> List[Dict[str, str]]:
        refined_split = []
        for sec in sections:
            refined_split.extend(self._split_long_section(sec))
        self.logger.info(f"步骤4-1：超长章节切分完成，共生成{len(refined_split)}个初始子Chunk")
    def _split_long_section(self,section: Dict[str, str]) -> List[Dict[str, str]]:
        content = section.get("content","")
        if len(content) <= DEFAULT_MAX_CONTENT_LENGTH:
            return [section]

        title = section.get("title","")
        prefix = f"{title}\n\n" if title else ""
        available_len = DEFAULT_MAX_CONTENT_LENGTH - len(prefix)
        if available_len < 0:
            self.logger.warning(f"章节标题过长，无法切分：{title[:20]}...")
            return [section]
        body = content
        if title and body.lstrip().startswith(title):
            body = body[body.find(title)+len(title):].lstrip()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=available_len,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
        )

        sub_sections = []
        for idx,chunk in enumerate(splitter.split_text(body), start=1):
            text = chunk.strip()
            if not text:
                continue

            full_text = (prefix+body).strip()

            sub_sections.append({
                "title": f"{title}-{idx}" if title else f"chunk-{idx}",
                "content": full_text,
                "parent_title":title,
                "part":idx,
                "file_title": section.get("file_title"),
            })

        self.self.logger.debug(f"超长章节切分完成：{title} → 生成{len(sub_sections)}个子Chunk")
        return sub_sections


    def _merge_short_sections(self):
        pass



if __name__ == "__main__":
    setup_logging()

    node = NodeDocumentSplit()
    md_path = r"D:\qdd\hello_RAG\processor\output\Aolynk CB304n Cable网桥 用户手册-5W100-整本手册\Aolynk CB304n Cable网桥 用户手册-5W100-整本手册_new.md"
    with open(md_path, 'r') as f:
        md_content = f.read()

    init_state = {
        "md_path": md_path,
        "md_content": md_content,
        "file_title": "Aolynk 用户手册"
    }

    result = node(init_state)
    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))
