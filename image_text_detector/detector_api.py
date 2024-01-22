import os
import asyncio
import logging
from argparse import Namespace
from typing import Tuple

from .image_text_detector import (
    TextDetector,
    set_main_logger,
)
from .args import parser
from .utils import (
    BASE_PATH,
    init_logging,
    get_logger,
    set_log_level,
    natural_sort,
)


class TextDetectorApi:
    """
    This class is used to run the text detection pipeline from a web API.
    """

    def __init__(self):
        self.detector = TextDetector()

    async def detect(self, args: Namespace):
        """
        Runs the text detection pipeline.
        """
        init_logging()
        try:
            set_log_level(level=logging.DEBUG if args.verbose else logging.INFO)
            logger = get_logger(args.mode)
            set_main_logger(logger)
            results = await self._detect(args)
            return results
        except KeyboardInterrupt:
            if not args or args.mode != 'web':
                print()
                return []
        except Exception as e:
            logger = get_logger(args.mode)
            logger.error(f'{e.__class__.__name__}: {e}',
                         exc_info=e if args and args.verbose else None)
            return []

    async def _detect(self, args: Namespace):
        args_dict = vars(args)

        if args.mode in ('demo', 'batch'):
            if not args.input:
                raise Exception('No input image was supplied. Use -i <image_path>')
            detector = TextDetector(args_dict)
            if args.mode == 'demo':
                if len(args.input) != 1 or not os.path.isfile(args.input[0]):
                    raise FileNotFoundError(
                        f'Invalid single image file path for demo mode: "{" ".join(args.input)}". Use `-m batch`.')
                dest = os.path.join(BASE_PATH, 'result/final.png')
                args.overwrite = True  # Do overwrite result/final.png file
                tag, result = await detector.detect_path(args.input[0], dest, args_dict)
                if tag:
                    result_dict = {
                        'status': 'success',
                        'result': result
                    }
                else:
                    result_dict = {
                        'status': 'failure',
                    }
                return [result_dict]
            else:  # batch
                dir = args.input[0]
                results = []
                if not os.path.isdir(dir):
                    raise NotADirectoryError(f'Invalid directory path: {dir}')
                for filename in natural_sort(os.listdir(dir)):
                    if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.webp'):
                        file_path = os.path.join(dir, filename)
                        tag, result = await detector.detect_path(file_path, "", args_dict)
                        if tag:
                            result_dict = {
                                'status': 'success',
                                'result': result
                            }
                        else:
                            result_dict = {
                                'status': 'failure',
                            }
                        results.append(result_dict)
                return results
