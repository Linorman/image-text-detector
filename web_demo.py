import gradio as gr
from image_text_detector.image_text_detector import TextDetector
import os

os.environ["no_proxy"] = "localhost,127.0.0.1,::1"


def detect(folder_path):
    # Initialize TextDetector with default parameters
    detector = TextDetector()
    # Run detection on all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.webp'):
            file_path = os.path.join(folder_path, filename)
            detector.detect_path(file_path)
    return "Detection completed"


with gr.Blocks() as demo:
    folder_path = gr.FileExplorer(label="Select a folder")
    output = gr.Textbox(label="Output")
    generate_btn = gr.Button("Detect")
    generate_btn.click(fn=detect, inputs=folder_path, outputs=output, api_name="detect")

demo.launch(share=True)
