import asyncio
import os
import json
import cv2
from argparse import Namespace

from image_text_detector import TextDetectorApi

clips_dir = '/home/ubuntu/image-text-detector/clips'
detector = TextDetectorApi()
dirs = os.listdir(clips_dir)
# batch mode
args_temp = Namespace(
    mode='batch',
    input=[],
    verbose=False,
)


def handle_clip(clip_path):
    # 将clip切成帧，保存在temp文件夹下
    # 创建temp文件夹
    print("Handling clip: ", clip_path)
    temp_path = os.path.join(clips_dir, 'temp')
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    cap = cv2.VideoCapture(clip_path)
    success, frame = cap.read()
    count = 0
    while success:
        cv2.imwrite(os.path.join(temp_path, 'frame%d.jpg' % count), frame)
        success, frame = cap.read()
        count += 1

    # batch mode
    args = Namespace(
        mode='batch',
        input=[temp_path],
        verbose=False,
        use_gpu=True,
    )
    detector = TextDetectorApi()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results = loop.run_until_complete(detector.detect(args))
    loop.close()
    if len(results) == 0:
        return None
    else:
        # 将temp文件夹下的帧删除
        for file in os.listdir(temp_path):
            os.remove(os.path.join(temp_path, file))
        os.rmdir(temp_path)
        return results


for dir in dirs:
    clip_path = os.path.join(clips_dir, dir)
    # 创建保存结果的json文件
    text_json_data = []
    mis_text_json_data = []
    error_json_data = []
    for file in os.listdir(clip_path):
        # 处理视频文件
        if file.endswith('.mp4'):
            clip_path = os.path.join(clip_path, file)
            results = handle_clip(clip_path)
            if results is not None:
                for result in results:
                    if result['status'] != 'success':
                        error_json_data.append(results)
                        break
                    else:
                        if result['result']['tag'] == 1:
                            continue
                        elif result['result']['tag'] == 2:
                            # 保存detect误检测的结果
                            result['result']['image'] = clip_path
                            mis_text_json_data.append(result['result'])
                        else:
                            # 保存有文字的检测结果
                            result['result']['image'] = clip_path
                            text_json_data.append(result['result'])
                            break
        else:
            continue
    # 保存结果
    text_json_path = os.path.join(clip_path, '_text.json')
    mis_text_json_path = os.path.join(clip_path, '_mis_text.json')
    error_json_path = os.path.join(clip_path, '_error.json')
    with open(text_json_path, 'w') as f:
        json.dump(text_json_data, f)
    with open(mis_text_json_path, 'w') as f:
        json.dump(mis_text_json_data, f)
    with open(error_json_path, 'w') as f:
        json.dump(error_json_data, f)

