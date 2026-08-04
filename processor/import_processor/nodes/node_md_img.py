# processor/import_processor/nodes/node_md_img.py
import base64
import json
import logging
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Tuple, List, Dict, Deque

from langchain.chat_models import init_chat_model

from processor.import_processor.base import BaseNode, setup_logging
from configs.lm_config import lm_config
from processor.import_processor.exceptions import StateFieldError, FileProcessingError
from processor.import_processor.state import ImportGraphState


class NodeMDImg(BaseNode):
    """
    MarkDown图片处理节点：多模态图片理解
    """

    name = "node_md_img"

    def process(self, state: ImportGraphState):
        # 获取内容
        md_content, md_path_obj, images_dir = self._setp_1_check_md(state)

        # 提取内容图片
        if not images_dir.exists():
            self.logger.info("无图片文件夹，跳过图片处理")
            return state

        target_images = self._step_2_scan_images(state, images_dir)
        # 生成摘要
        if not target_images:
            self.logger.info("无可处理图片，跳过生成摘要")
            return state
        self._step_3_generate_summaries(md_path_obj.stem, target_images)
        # 替换md图片地址为minio地址,alt写摘要内容
        # 备份新的md
        return state

    def _setp_1_check_md(self, state: ImportGraphState) -> Tuple[str, Path, Path]:
        """
                从全局状态中提取并初始化MD处理所需核心数据
                :param state: 流程全局状态对象
                :return: 元组(MD文件内容, MD文件路径, 图片文件夹路径)
                :raise FileProcessingError: 当状态中无有效MD文件路径时抛出
        """
        md_path = state.get("md_path")
        md_content = state["md_content"]

        if not md_path:
            raise StateFieldError(field_name="pdf_path", expected_type=str)

        md_path_obj = Path(md_path)

        if not md_path_obj.exists():
            raise FileProcessingError(message=f"{md_path_obj.name}不存在")

        images_dir = md_path_obj.parent / "images"

        return md_content, md_path_obj, images_dir

    def _step_2_scan_images(self, state: ImportGraphState, images_dir: Path) -> List[Tuple[str, str, Tuple[str, str]]]:
        """
                扫描图片文件夹，过滤出「支持格式+MD中实际引用」的图片，组装处理元数据
                :param md_content: MD文件完整内容
                :param images_dir: 图片文件夹路径对象
                :return: 待处理图片列表，每个元素为(图片文件名, 图片完整路径, 图片上下文)元组
        """
        target_images = []
        for image_file in os.listdir(images_dir):
            file_ext = os.path.splitext(image_file)[1]
            if file_ext not in self.config.extensions:
                self.logger.warning(f"图片格式不支持，跳过：{image_file}")
                continue

            img_path = str(images_dir / image_file)

            context = self._find_image_in_md(md_content, image_file)

            if not context:
                self.logger.warning(f"图片未在MD中引用，跳过处理：{image_file}")
                continue

            target_images.append((image_file, img_path, context))
        return target_images

    def _find_image_in_md(self, md_content: str, image_file: str, context_len: int = 100) -> Tuple[str, str]:
        """
        查找MD内容中指定图片的所有引用位置，并返回每个位置的上下文文本
        :param md_content: MD文件完整内容
        :param image_file: 图片文件名（含后缀）
        :param context_len: 上下文截取长度，默认前后各100字符
        :return: 每个图片的(上文, 下文)元组，无匹配则返回None
        """
        pattern = re.compile(r"!\[.*?\]\(.*?") + re.escape(image_file) + r".*?\)"

        match = pattern.search(md_content)
        if not match:
            return None

        start, end = match.span()
        pre_text = md_content[max(0, start - context_len):end]
        post_text = md_content[end:min(len(md_content), end + context_len)]

        return pre_text, post_text

    def _step_3_generate_summaries(self, doc_stem: str, target_images: List[Tuple[str, str, Tuple[str, str]]]) -> Dict[str, str]:
        """
                步骤3：批量为待处理图片生成内容摘要，带API速率限制防止触发大模型限流
                :param doc_stem: 文档文件名（不含后缀），作为大模型prompt上下文
                :param targets: 待处理图片列表，元素为(图片文件名, 图片完整路径, 图片上下文)
                :param requests_per_minute: 每分钟最大API请求数，默认9次（按大模型限制调整）
                :return: 图片摘要字典，键：图片文件名，值：图片内容摘要
        """
        summaries = {}

        request_deque = deque()
        for img_file, image_path, context in target_images:
            self._apply_api_rate_limit(request_deque)

            summaries[img_file] = self._summarize_image(image_path, root_folder=doc_stem, image_content=context)
        return summaries

    def _apply_api_rate_limit(self, request_times: Deque[float], max_requests: int = 100, window_seconds: int = 60) -> None:
        """
        通用滑动窗口API速率限制器（抽离为公共工具）
        核心逻辑：维护请求时间戳双端队列，窗口内请求数超上限则自动等待，防止触发第三方API限流
        :param request_times: 存储请求时间戳的双端队列，需外部初始化（全局/单例），跨调用复用
        :param max_requests: 速率限制窗口内的最大允许请求次数
        :param window_seconds: 速率限制滑动窗口时长，默认60秒（1分钟）
        :return: None，超出限制时会阻塞等待
        """
        current_time = time.time()

        while request_times and current_time - request_times[0] > max_requests:
            request_times.popleft()

        if len(request_times) >= max_requests:
            sleep_duration = window_seconds - (current_time - request_times[0])
            if sleep_duration > 0:
                logging.getLogger().info(
                    f"触发API速率限制，窗口{window_seconds}秒内最多{max_requests}次，需等待：{sleep_duration:.2f} 秒")
                time.sleep(sleep_duration)
            current_time = time.time()
            while request_times and current_time - request_times[0] >= window_seconds:
                request_times.popleft()

        request_times.append(current_time)
        logging.getLogger().info(f"API请求时间戳已记录，当前{window_seconds}秒窗口内请求数：{len(request_times)}")

    def _summarize_image(self, image_path: Path, root_folder: str, image_content: str) -> str:
        """
           调用多模态大模型总结图片内容。

           参数：
           - image_path: 图片本地路径。
           - root_folder: 文档所属文件夹名（提供更多上下文）。
           - image_content: 图片在文档中的上下文 (前文, 后文)。
        """
        with open(image_path, 'rb') as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        try:
            chat_model = init_chat_model(
                model=lm_config.model_name,
                model_provider="openai",
                api_key=lm_config.api_key,
                baseUrl=lm_config.baseUrl,
                temperature=lm_config.temperature,
            )

            messages = [
                {
                    "role":"user",
                    "content":[
                        {
                            "type":"text",
                            "text":f"""这是"{root_folder}"文件中的一张图片，图片上文部分为"{image_content[0]}"，下文部分为"{image_content[1]}"，请用中文简要总结这张图片的内容，用于 Markdown 图片标题。"""
                        },
                        {
                            "type":"image",
                            "image_url":{
                                "url":f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]

            response = chat_model.invoke(messages)
            return response.content.strip().replace('\n', '')
        except Exception as e:
            self.logger.error(f"图像总结失败：{image_path}, 错误{e}")
            return "图片描述"


if __name__ == "__main__":
    setup_logging()

    md_path = r"D:\qdd\hello_RAG\processor\output\Aolynk CB304n Cable网桥 用户手册-5W100-整本手册\Aolynk CB304n Cable网桥 用户手册-5W100-整本手册.md"
    with open(md_path, 'r') as f:
        md_content = f.read()

    state = {
        "md_path": md_path,
        "md_content": md_content,
    }
    node = NodeMDImg()
    result = node(state)
    logging.Logger.info(json.dumps(result, ensure_ascii=False, indent=4))
