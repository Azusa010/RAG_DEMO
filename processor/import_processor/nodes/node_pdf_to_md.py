import logging
import shutil
import time
import zipfile
from pathlib import Path

import requests

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, FileProcessingError, ConfigurationError, \
    PdfConversionError
from processor.import_processor.state import ImportGraphState
from configs.mineru_config import mineru_config


class NodePDFToMD(BaseNode):
    name = 'node_pdf_to_md'

    def process(self, state: ImportGraphState):
        # 检验文档
        pdf_path_obj, file_path_obj = self.step_1_validate_pdf(state)
        # 上传文档到minerU 并轮询获得结果
        zip_url = self.step_2_upload_and_pull(pdf_path_obj)
        # 下载与解压
        md_path = self._step_3_download_and_extract(zip_url,file_path_obj,pdf_path_obj.stem)

        with open(md_path, "rb") as f:
            md_content = f.read()
        state["md_path"] = str(md_path)
        state["md_content"] = md_content

        return state

    def step_1_validate_pdf(self, state: ImportGraphState):
        pdf_path = state.get("pdf_path")
        file_path = state.get("file_dir")
        if not pdf_path:
            raise StateFieldError(field_name="pdf_path", expected_type=str)
        if not file_path:
            raise StateFieldError(field_name="pdf_path", expected_type=str)

        pdf_path_obj = Path(pdf_path)
        file_path_obj = Path(file_path)

        if not pdf_path_obj.exists():
            raise FileProcessingError(message=f"{pdf_path_obj.name}不存在")

        if not file_path_obj.exists():
            self.logger.info(f"输出目录不存在，自动创建：{file_path_obj.absolute()}")
            file_path_obj.mkdir(parents=True, exist_ok=True)

        return pdf_path_obj, file_path_obj

    def step_2_upload_and_pull(self, pdf_path_obj: Path):
        if not mineru_config.base_url:
            raise ConfigurationError(message="MinerU配置缺失：请在 .env 文件中正确配置 MINERU_BASE_URL 参数")
        if not mineru_config.api_token:
            raise ConfigurationError(message="MinerU配置缺失：请在 .env 文件中正确配置 MINERU_API_TOKEN 参数")

        token = mineru_config.api_token
        url = f"{mineru_config.base_url}/file-urls/batch"
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        data = {
            "files": [
                {"name": pdf_path_obj.name}
            ],
            "model_version": "vlm"
        }
        file_path = [pdf_path_obj]

        response = requests.post(url, headers=header, json=data)
        if response.status_code != 200:
            raise PdfConversionError(message=f"获取上传链接响应失败：状态码：{response.status_code}，响应结果：{response}")

        result = response.json()

        if result["code"] != 0:
            raise PdfConversionError(f"获取上传链接失败：返回数据：{result}")

        urls = result["data"]["file_urls"]
        batch_id = result["data"]["batch_id"]
        print('batch_id:{},urls:{}'.format(batch_id, urls))
        #
        for i in range(0, len(urls)):
            with open(pdf_path_obj, "rb") as f:
                res_upload = requests.put(urls[i], data=f)
                if res_upload.status_code != 200:
                    raise PdfConversionError(f"文件上传失败：状态码：{res_upload.status_code}，响应结果：{res_upload}")

                self.logger.info(f"文件{pdf_path_obj.name}上传成功！")

        # 批量获取任务结果
        pull_url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
        res = requests.get(pull_url,headers=header)
        start_time = time.time()
        timeout_seconds = 60
        pull_interval = 3

        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                raise TimeoutError(f"【任务轮询】超时！任务处理超{timeout_seconds}秒，batch_id：{batch_id}")

            try:
                res_pull = requests.get(pull_url, headers=header,timeout=10)
            except Exception as e:
                self.logger.warning(f"【任务轮询】网络请求异常，{pull_interval}秒后重试：{str(e)}，bactch_id：{batch_id}")
                time.sleep(pull_interval)
                continue

            if res_pull.status_code != 200:
                raise    PdfConversionError(f"【任务轮询】HTTP请求失败，状态码：{res_pull.status_code}，响应内容：{res_pull}")

            pull_data = res_pull.json()
            if pull_data["code"] != 0:
                raise PdfConversionError(f"【任务轮询】业务错误，返回数据：{pull_data}")

            extracted_result = pull_data["data"]["extract_result"]

            result_item = extracted_result[0]
            result_state = result_item["state"]

            if result_state == "done":
                self.logger.info(f"【任务轮询】解析任务完成！总耗时{int(elapsed_time)}s，bactch_id：{batch_id}")

                full_zip_url = result_item["full_zip_url"]
                self.logger.info(f"【任务轮询】返回ZIP包下载链接：{full_zip_url}，bactch_id：{batch_id}")

                return full_zip_url
            elif result_state == "failed":
                err_msg = result_item.get("err_msg", "未知错误，无具体信息")
                raise PdfConversionError(f"【任务轮询】解析任务失败！batch_id：{batch_id}，错误信息：{err_msg}")

            else:
                self.logger.info(
                    f"【任务轮询】处理中... 已耗时{int(elapsed_time)}s，状态：{result_state}， batch_id：{batch_id}")
                time.sleep(pull_interval)

    def _step_3_download_and_extract(self,zip_url:str,output_dir_obj:Path,pdf_stem:str)-> str:
        # 1、下载ZIP包
        self.logger.info(f"【ZIP下载】开始下载ZIP包：{zip_url} ...")
        res=  requests.get(zip_url)

        if res.status_code != 200:
            raise   RuntimeError(f"【ZIP下载】ZIP包下载失败：状态码：{res.status_code}，响应结果：{res}")

        zip_save_path = output_dir_obj / f"{pdf_stem}_output.zip"
        with open(zip_save_path, "wb") as f:
            f.write(res.content)
        self.logger.info(f"【ZIP下载】ZIP包下载成功：保存路径：{zip_save_path}")

        extract_target_dir = output_dir_obj / pdf_stem
        if extract_target_dir.exists():
            shutil.rmtree(extract_target_dir)
        self.logger.info(f"【ZIP解压】已清空旧的解压目录：{extract_target_dir}")

        extract_target_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"【ZIP解压】开始解压ZIP包：{output_dir_obj} ...")
        with zipfile.ZipFile(zip_save_path, "r") as zip_ref:
            zip_ref.extractall(extract_target_dir)
        self.logger.info(f"【ZIP解压】ZIP解压完成，解压目录：{extract_target_dir}")

        self.logger.info(f"【MD重命名】找到MinerU生成的full.md文件")
        target_md_file = extract_target_dir / "full.md"
        self.logger.info(f"【MD重命名】开始将full.md文件进行重命名")
        new_md_path = target_md_file.with_name(f"{pdf_stem}.md")
        target_md_file.rename(new_md_path)
        self.logger.info(f"【MD重命名】重命名成功，文件名：{pdf_stem}.md")

        return str(new_md_path.absolute())

if __name__ == "__main__":
    setup_logging()

    state = {
        "pdf_path": r"D:\qdd\hello_RAG\processor\doc\Aolynk CB304n Cable网桥 用户手册-5W100-整本手册.pdf",
        "file_dir": r"D:\qdd\hello_RAG\processor\output"
    }

    node = NodePDFToMD()
    result = node(state)
    logging.info(result)
