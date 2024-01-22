# Image-Text-Detector

## Introduction
This is a simple project to detect text in images. We use it simply to detect text in images and don't use it for OCR though it can do it.
The project is based on [manga-image-translator](https://github.com/zyddnys/manga-image-translator.git).

## Usage
### Install
```bash
pip install -r requirements.txt
# For Linux or Macos Users
pip install git+https://github.com/kodalli/pydensecrf.
# For Windows Users
# Before you start the pip install, first install Microsoft C++ Build Tools.
# If you have trouble installing pydensecrf with the command above you can install the pre-compiled wheels with pip install https://www.lfd.uci.edu/~gohlke/pythonlibs/#_pydensecrf.
```

### Run
```bash
python -m image_text_detector --verbose --mode [your_mode] -i [your_image_path]
```
You can use `python -m image_text_detector --help` to see the help message (though some of parameters are not supported yet).

If you want to use the model in your own project, you can use the following code:
```python
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
```
In args, you can set different parameters to control the behavior of the detector. And the `results` is a list of dict.
