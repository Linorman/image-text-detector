import logging
import os
import re
from typing import List, Union, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .args import DEFAULT_ARGS
from .colorization import dispatch as dispatch_colorization, prepare as prepare_colorization
from .detection import dispatch as dispatch_detection, prepare as prepare_detection
from .ocr import dispatch as dispatch_ocr, prepare as prepare_ocr
from .upscaling import dispatch as dispatch_upscaling, prepare as prepare_upscaling
from .utils import (
    BASE_PATH,
    ModelWrapper,
    Context,
    load_image,
    replace_prefix,
    rgb2hex,
    get_color_name,
    natural_sort,
    save_image_detect_result
)

# Will be overwritten by __main__.py if module is being run directly (with python -m)
logger = logging.getLogger('image_text_detector')


def set_main_logger(l):
    global logger
    logger = l


class TranslationInterrupt(Exception):
    """
    Can be raised from within a progress hook to prematurely terminate
    the translation.
    """
    pass


class TextDetectorOptimize:

    def __init__(self, params: dict = None):
        self.device = None
        self._gpu_limited_memory = None
        self.ignore_errors = None
        self.verbose = None
        self._progress_hooks = []
        self._add_logger_hook()

        params = params or {}
        self.parse_init_params(params)
        self.result_sub_folder = ''

        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
        # in PyTorch 1.12 and later.
        torch.backends.cuda.matmul.allow_tf32 = True

        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = True

    def parse_init_params(self, params: dict):
        self.verbose = params.get('verbose', False)
        self.ignore_errors = params.get('ignore_errors', False)
        # check mps for apple silicon or cuda for nvidia
        device = 'mps' if torch.backends.mps.is_available() else 'cuda'
        self.device = device if params.get('use_gpu', False) else 'cpu'
        self._gpu_limited_memory = params.get('use_gpu_limited', False)
        if self._gpu_limited_memory and not self.using_gpu:
            self.device = device
        if self.using_gpu and (not torch.cuda.is_available() and not torch.backends.mps.is_available()):
            raise Exception(
                'CUDA or Metal compatible device could not be found in torch whilst --use-gpu args was set.\n''Is the '
                'correct pytorch version installed? (See https://pytorch.org/)')
        if params.get('model_dir'):
            ModelWrapper._MODEL_DIR = params.get('model_dir')

        os.environ['INPAINTING_PRECISION'] = params.get('inpainting_precision', 'fp32')

    @property
    def using_gpu(self):
        return self.device.startswith('cuda') or self.device == 'mps'

    async def detect_path(self, path: str, dest: str = None, params: dict = None) -> Tuple[bool, dict]:
        """
        Translates an image or folder (recursively) specified through the path.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        path = os.path.abspath(os.path.expanduser(path))
        dest = os.path.abspath(os.path.expanduser(dest)) if dest else ''
        params = params or {}

        # Handle format
        file_ext = params.get('format')
        if params.get('save_quality', 100) < 100:
            if not params.get('format'):
                file_ext = 'jpg'
            elif params.get('format') != 'jpg':
                raise ValueError('--save-quality of lower than 100 is only supported for .jpg files')

        if os.path.isfile(path):
            # Determine destination file path
            if not dest:
                # Use the same folder as the source
                p, ext = os.path.splitext(path)
                _dest = f'{p}-translated.{file_ext or ext[1:]}'
            elif not os.path.basename(dest):
                p, ext = os.path.splitext(os.path.basename(path))
                # If the folders differ use the original filename from the source
                if os.path.dirname(path) != dest:
                    _dest = os.path.join(dest, f'{p}.{file_ext or ext[1:]}')
                else:
                    _dest = os.path.join(dest, f'{p}-translated.{file_ext or ext[1:]}')
            else:
                p, ext = os.path.splitext(dest)
                _dest = f'{p}.{file_ext or ext[1:]}'
            tag, ctx = await self.detect_file(path, _dest, params)
            result = {
                'image': ctx.input.filename,
                'detect_result': ctx.detect_result,
                'tag': ctx.detect_tag,
            }
            return tag, result

        elif os.path.isdir(path):
            # Determine destination folder path
            if path[-1] == '\\' or path[-1] == '/':
                path = path[:-1]
            _dest = dest or path + '-translated'
            if os.path.exists(_dest) and not os.path.isdir(_dest):
                raise FileExistsError(_dest)

            detected_count = 0
            for root, subdirs, files in os.walk(path):
                files = natural_sort(files)
                dest_root = replace_prefix(root, path, _dest)
                os.makedirs(dest_root, exist_ok=True)
                for f in files:
                    if f.lower() == '.thumb':
                        continue

                    file_path = os.path.join(root, f)
                    output_dest = replace_prefix(file_path, path, _dest)
                    p, ext = os.path.splitext(output_dest)
                    output_dest = f'{p}.{file_ext or ext[1:]}'
                    tag, ctx = await self.detect_file(path, _dest, params)
                    if tag:
                        detected_count += 1
            if detected_count == 0:
                logger.info('No further untranslated files found. Use --overwrite to write over existing translations.')
            else:
                logger.info(f'Done. Detected {detected_count} image{"" if detected_count == 1 else "s"}')
            result = {
                'image': ctx.input,
                'detect_result': ctx.detect_result,
                'tag': ctx.detect_tag,
            }
            return tag, result

    async def detect_file(self, path: str, dest: str, params: dict) -> Tuple[bool, Context]:
        logger.info(f'Handling: "{path}"')

        # Turn dict to context to make values also accessible through params.<property>
        params = params or {}
        ctx = Context(**params)
        self._preprocess_params(ctx)

        attempts = 0
        while ctx.attempts == -1 or attempts < ctx.attempts + 1:
            if attempts > 0:
                logger.info(f'Retrying translation! Attempt {attempts}'
                            + (f' of {ctx.attempts}' if ctx.attempts != -1 else ''))
            try:
                return await self._detect_file(path, dest, ctx)

            except TranslationInterrupt:
                break
            except Exception as e:
                await self._report_progress('error', True)
                if not self.ignore_errors and not (ctx.attempts == -1 or attempts < ctx.attempts):
                    raise
                else:
                    logger.error(f'{e.__class__.__name__}: {e}',
                                 exc_info=e if self.verbose else None)
            attempts += 1
        return False, ctx

    async def _detect_file(self, path: str, dest: str, ctx: Context) -> Tuple[bool, Context]:
        # Treat as image
        try:
            img = Image.open(path)
            img.verify()
            img = Image.open(path)
        except Exception:
            logger.warning(f'Failed to open image: {path}')
            return False, ctx

        ctx = await self.detect(img, ctx)

        return True, ctx

    async def detect(self, image: Image.Image, params: Union[dict, Context] = None) -> Context:
        """
        Translates a PIL image from a manga. Returns dict with result and intermediates of translation.
        Default params are taken from args.py.

        ```py
        translation_dict = await translator.translate(image)
        result = translation_dict.result
        ```
        """
        # TODO: Take list of images to speed up batch processing

        if not isinstance(params, Context):
            params = params or {}
            ctx = Context(**params)
            self._preprocess_params(ctx)
        else:
            ctx = params

        ctx.input = image
        ctx.result = None

        # preload and download models (not strictly necessary, remove to lazy load)
        logger.info('Loading models')
        if ctx.upscale_ratio:
            await prepare_upscaling(ctx.upscaler)
        await prepare_detection(ctx.detector)
        await prepare_ocr(ctx.ocr, self.device)
        if ctx.colorizer:
            await prepare_colorization(ctx.colorizer)
        # detect
        return await self._detect(ctx)

    def _preprocess_params(self, ctx: Context):
        # params auto completion
        # TODO: Move args into ctx.args and only calculate once, or just copy into ctx
        for arg in DEFAULT_ARGS:
            ctx.setdefault(arg, DEFAULT_ARGS[arg])

        if 'direction' not in ctx:
            if ctx.force_horizontal:
                ctx.direction = 'h'
            elif ctx.force_vertical:
                ctx.direction = 'v'
            else:
                ctx.direction = 'auto'
        if 'alignment' not in ctx:
            if ctx.align_left:
                ctx.alignment = 'left'
            elif ctx.align_center:
                ctx.alignment = 'center'
            elif ctx.align_right:
                ctx.alignment = 'right'
            else:
                ctx.alignment = 'auto'
        if ctx.prep_manual:
            ctx.renderer = 'none'
        ctx.setdefault('renderer', 'manga2eng' if ctx.manga2eng else 'default')

        if ctx.filter_text:
            ctx.filter_text = re.compile(ctx.filter_text)

    async def _detect(self, ctx: Context) -> Context:
        image_filename = ctx.input.filename
        # -- Colorization
        if ctx.colorizer:
            await self._report_progress('colorizing')
            ctx.img_colorized = await self._run_colorizer(ctx)
        else:
            ctx.img_colorized = ctx.input

        # -- Upscaling
        # The default text detector doesn't work very well on smaller images, might want to
        # consider adding automatic upscaling on certain kinds of small images.
        if ctx.upscale_ratio:
            await self._report_progress('upscaling')
            ctx.upscaled = await self._run_upscaling(ctx)
        else:
            ctx.upscaled = ctx.img_colorized

        ctx.img_rgb, ctx.img_alpha = load_image(ctx.upscaled)

        # -- Detection
        await self._report_progress('detection')
        ctx.textlines, ctx.mask_raw, ctx.mask = await self._run_detection(ctx)
        if self.verbose:
            cv2.imwrite(self._result_path('mask_raw.png'), ctx.mask_raw)

        if not ctx.textlines:
            await self._report_progress('skip-no-regions', True)
            # If no text was found result is intermediate image product
            ctx.result = ctx.upscaled
            if ctx.json_path is None:
                ctx.detect_result = "no text detected"
                ctx.detect_tag = 1
                await save_image_detect_result(image_filename, "no text detected", tag=1)
            else:
                ctx.detect_result = "no text detected"
                ctx.detect_tag = 1
                await save_image_detect_result(image_filename, "no text detected", tag=1, json_path=ctx.json_path)
            return ctx
        if self.verbose:
            img_bbox_raw = np.copy(ctx.img_rgb)
            for txtln in ctx.textlines:
                cv2.polylines(img_bbox_raw, [txtln.pts], True, color=(255, 0, 0), thickness=2)
            cv2.imwrite(self._result_path('bboxes_unfiltered.png'), cv2.cvtColor(img_bbox_raw, cv2.COLOR_RGB2BGR))

        # -- OCR
        await self._report_progress('ocr')
        ctx.textlines = await self._run_ocr(ctx)
        if not ctx.textlines:
            await self._report_progress('skip-no-text', True)
            # If no text was found result is intermediate image product
            ctx.result = ctx.upscaled
            if ctx.json_path is None:
                ctx.detect_result = "no text ocr"
                ctx.detect_tag = 2
                await save_image_detect_result(image_filename, "no text ocr", tag=2)
            else:
                ctx.detect_result = "no text ocr"
                ctx.detect_tag = 2
                await save_image_detect_result(image_filename, "no text ocr", tag=2, json_path=ctx.json_path)
            return ctx

        if ctx.json_path is None:
            ctx.detect_result = "some text detected"
            ctx.detect_tag = 0
            await save_image_detect_result(image_filename, "some text detected", tag=0)
        else:
            ctx.detect_result = "some text detected"
            ctx.detect_tag = 0
            await save_image_detect_result(image_filename, "some text detected", tag=0, json_path=ctx.json_path)

        if not ctx.text_regions:
            await self._report_progress('error-translating', True)
            ctx.result = ctx.upscaled
            return ctx
        elif ctx.text_regions == 'cancel':
            await self._report_progress('cancelled', True)
            ctx.result = ctx.upscaled
            return ctx

        return ctx

    async def _run_colorizer(self, ctx: Context):
        return await dispatch_colorization(ctx.colorizer, device=self.device, image=ctx.input, **ctx)

    async def _run_upscaling(self, ctx: Context):
        return (await dispatch_upscaling(ctx.upscaler, [ctx.img_colorized], ctx.upscale_ratio, self.device))[0]

    async def _run_detection(self, ctx: Context):
        return await dispatch_detection(ctx.detector, ctx.img_rgb, ctx.detection_size, ctx.text_threshold,
                                        ctx.box_threshold,
                                        ctx.unclip_ratio, ctx.det_invert, ctx.det_gamma_correct, ctx.det_rotate,
                                        ctx.det_auto_rotate,
                                        self.device, self.verbose)

    async def _run_ocr(self, ctx: Context):
        textlines = await dispatch_ocr(ctx.ocr, ctx.img_rgb, ctx.textlines, ctx, self.device, self.verbose)

        new_textlines = []
        for textline in textlines:
            if textline.text.strip():
                if ctx.font_color_fg:
                    textline.fg_r, textline.fg_g, textline.fg_b = ctx.font_color_fg
                if ctx.font_color_bg:
                    textline.bg_r, textline.bg_g, textline.bg_b = ctx.font_color_bg
                new_textlines.append(textline)
        return new_textlines

    def _result_path(self, path: str) -> str:
        """
        Returns path to result folder where intermediate images are saved when using verbose flag
        or web mode input/result images are cached.
        """
        return os.path.join(BASE_PATH, 'result', self.result_sub_folder, path)

    def add_progress_hook(self, ph):
        self._progress_hooks.append(ph)

    async def _report_progress(self, state: str, finished: bool = False):
        for ph in self._progress_hooks:
            await ph(state, finished)

    def _add_logger_hook(self):
        # TODO: Pass ctx to logger hook
        LOG_MESSAGES = {
            'upscaling': 'Running upscaling',
            'detection': 'Running text detection',
            'ocr': 'Running ocr',
            'mask-generation': 'Running mask refinement',
            'colorizing': 'Running colorization',
            'downscaling': 'Running downscaling',
        }
        LOG_MESSAGES_SKIP = {
            'skip-no-regions': 'No text regions! - Skipping',
            'skip-no-text': 'No text regions with text! - Skipping',
            'cancelled': 'Image translation cancelled',
        }
        LOG_MESSAGES_ERROR = {
            # 'error-lang':           'Target language not supported by chosen translator',
        }

        async def ph(state, finished):
            if state in LOG_MESSAGES:
                logger.info(LOG_MESSAGES[state])
            elif state in LOG_MESSAGES_SKIP:
                logger.warn(LOG_MESSAGES_SKIP[state])
            elif state in LOG_MESSAGES_ERROR:
                logger.error(LOG_MESSAGES_ERROR[state])

        self.add_progress_hook(ph)

    def _save_text_to_file(self, image_path: str, ctx: Context):
        cached_colors = []

        def identify_colors(fg_rgb: List[int]):
            idx = 0
            for rgb, _ in cached_colors:
                # If similar color already saved
                if abs(rgb[0] - fg_rgb[0]) + abs(rgb[1] - fg_rgb[1]) + abs(rgb[2] - fg_rgb[2]) < 50:
                    break
                else:
                    idx += 1
            else:
                cached_colors.append((fg_rgb, get_color_name(fg_rgb)))
            return idx + 1, cached_colors[idx][1]

        s = f'\n[{image_path}]\n'
        for i, region in enumerate(ctx.text_regions):
            fore, back = region.get_font_colors()
            color_id, color_name = identify_colors(fore)

            s += f'\n-- {i + 1} --\n'
            s += f'color: #{color_id}: {color_name} (fg, bg: {rgb2hex(*fore)} {rgb2hex(*back)})\n'
            s += f'text:  {region.text}\n'
            s += f'trans: {region.translation}\n'
            for line in region.lines:
                s += f'coords: {list(line.ravel())}\n'
        s += '\n'

        text_output_file = ctx.text_output_file
        if not text_output_file:
            text_output_file = os.path.splitext(image_path)[0] + '_translations.txt'

        with open(text_output_file, 'a', encoding='utf-8') as f:
            f.write(s)
