import asyncio
from argparse import Namespace

from image_text_detector import TextDetectorApi

if __name__ == '__main__':
    # batch mode
    args = Namespace(
        mode='batch',
        input=['F:\\Lenevo\\Desktop\\image-text-detector\\casts'],
        verbose=False,
    )
    detector = TextDetectorApi()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results = loop.run_until_complete(detector.detect(args))
    print(results)
    loop.close()
