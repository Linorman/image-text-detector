import asyncio
from argparse import Namespace

from image_text_detector import TextDetectorApi

if __name__ == '__main__':
    # demo mode
    args = Namespace(
        mode='demo',
        input=['F:\\Lenevo\\Desktop\\image-text-detector\\casts\\cast1.png'],
        verbose=False,
    )
    detector = TextDetectorApi()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results = loop.run_until_complete(detector.detect(args))
    print(results)
