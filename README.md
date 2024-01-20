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
