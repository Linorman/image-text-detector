import json
import logging
import os
import time

import colorama

from .generic import replace_prefix

ROOT_TAG = 'manga-translator'


class Formatter(logging.Formatter):
    def formatMessage(self, record: logging.LogRecord) -> str:
        if record.levelno >= logging.ERROR:
            self._style._fmt = f'{colorama.Fore.RED}%(levelname)s:{colorama.Fore.RESET} [%(name)s] %(message)s'
        elif record.levelno >= logging.WARN:
            self._style._fmt = f'{colorama.Fore.YELLOW}%(levelname)s:{colorama.Fore.RESET} [%(name)s] %(message)s'
        elif record.levelno == logging.DEBUG:
            self._style._fmt = '[%(name)s] %(message)s'
        else:
            self._style._fmt = '[%(name)s] %(message)s'
        return super().formatMessage(record)


class Filter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Try to filter out logs from imported modules
        if not record.name.startswith(ROOT_TAG):
            return False
        # Shorten the name
        record.name = replace_prefix(record.name, ROOT_TAG + '.', '')
        return super().filter(record)


root = logging.getLogger(ROOT_TAG)


def init_logging():
    logging.basicConfig(level=logging.INFO)
    for h in logging.root.handlers:
        h.setFormatter(Formatter())
        h.addFilter(Filter())


def set_log_level(level):
    root.setLevel(level)


def get_logger(name: str):
    return root.getChild(name)


file_handlers = {}


def add_file_logger(path: str):
    if path in file_handlers:
        return
    file_handlers[path] = logging.FileHandler(path, encoding='utf8')
    logging.root.addHandler(file_handlers[path])


def remove_file_logger(path: str):
    if path in file_handlers:
        logging.root.removeHandler(file_handlers[path])
        file_handlers[path].close()
        del file_handlers[path]


async def save_image_detect_result(image_path, result, tag=0, json_path='.'):
    """
    以json格式保存文字识别结果
    Args:
        json_path: json文件保存的位置
        image_path: 图片保存的位置
        result: 图片检测的结果
        tag: 结果类型
             - 0: 有文字
             - 1: 无文字-detection未检测到
             - 2: 无文字-ocr未检测到
    Returns:
    """
    json_path = os.path.join(json_path, 'result.json')

    if not os.path.exists(json_path):
        with open(json_path, 'w') as f:
            json.dump([], f, indent=4, ensure_ascii=False)

    with open(json_path, 'r') as f:
        # 判断json文件是否合法
        try:
            log_json = json.load(f)
        except json.decoder.JSONDecodeError:
            log_json = []
    result_json = {
        'image_path': image_path,
        'result': result,
        'tag': tag,
        'time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    }
    log_json.append(result_json)
    with open(json_path, 'w') as f:
        json.dump(log_json, f, indent=4, ensure_ascii=False)