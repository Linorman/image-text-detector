import os
import asyncio
import logging
from argparse import Namespace

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


async def dispatch(args: Namespace):
    args_dict = vars(args)

    logger.info(f'Running in {args.mode} mode')

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
            tag, ctx = await detector.detect_path(args.input[0], dest, args_dict)
        else:  # batch
            dest = args.dest
            dir = args.input[0]
            if not os.path.isdir(dir):
                raise NotADirectoryError(f'Invalid directory path: {dir}')
            for filename in natural_sort(os.listdir(dir)):
                if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.webp'):
                    file_path = os.path.join(dir, filename)
                    tag, ctx = await detector.detect_path(file_path, dest, args_dict)


if __name__ == '__main__':
    args = None
    init_logging()
    try:
        args = parser.parse_args()
        set_log_level(level=logging.DEBUG if args.verbose else logging.INFO)
        logger = get_logger(args.mode)
        set_main_logger(logger)
        if args.mode != 'web':
            logger.debug(args)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(dispatch(args))
    except KeyboardInterrupt:
        if not args or args.mode != 'web':
            print()
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}',
                     exc_info=e if args and args.verbose else None)
