import colorama
from dotenv import load_dotenv

colorama.init(autoreset=True)
load_dotenv()

from .image_text_detector import *
