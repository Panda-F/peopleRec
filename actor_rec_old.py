import time
from functools import wraps

import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
from mmdet.apis import DetInferencer
from mmpretrain import ImageClassificationInferencer


def timer(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        begin_time = time.perf_counter()
        result = func(*args, **kwargs)
        start_time = time.perf_counter()
        print('func:%r args:[%r, %r] took: %2.4f s' % (func.__name__, args, kwargs, start_time - begin_time))
        return result

    return wrap


def draw_bbox(img, bboxes, labels):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font='/home/fangzhou/下载/SimHei.ttf', size=18)
    label_dict = {"guodegang": "郭德纲", "yuqian": "于谦"}
    colors = {"guodegang": (0, 0, 255), "yuqian": (0, 255, 0)}
    for i, box in enumerate(bboxes):
        x1, y1, x2, y2 = box
        color = colors[labels[i]]
        draw.rectangle([x1, y1, x2, y2], outline=color)
        draw.text(xy=(x1, y1), text=label_dict[labels[i]], fill=color, font=font)
    return np.asarray(img)


class PeopleRec:
    def __init__(self):
        self.cls_inferencer = ImageClassificationInferencer(
            model='/home/fangzhou/Project/mmpretrain/xiangsheng_cls/xiangsheng_cls.py',
            pretrained='/home/fangzhou/Checkpoints/xiangshengcls/best_accuracy_top1_epoch_6.pth',
            device='cuda')
        self.det_inferencer = DetInferencer(model='rtmdet_tiny_8xb32-300e_coco',
                                            weights='/home/fangzhou/Checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth')

    def read_video(self, video_path):
        video_frames = []
        capture = cv2.VideoCapture(video_path)
        fps = capture.get(cv2.CAP_PROP_FPS)
        size = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        if capture.isOpened():
            while True:
                ret, img = capture.read()
                video_frames.append(img)
                if not ret:
                    break
        # video_frames = np.asarray(video_frames, dtype=object)
        return fps, size, video_frames

    @timer
    def infer(self, video_path, output_path="output.mp4"):
        fps, size, video_frames = self.read_video(video_path)
        videoWriter = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc("M", "J", "P", "G"),  # 编码器
            fps,
            size
        )
        for i, frame in enumerate(video_frames):
            print(f"process frame {i}")
            det_res = self.det_inferencer(frame)
            bboxes = []
            scores = []
            labels = []
            try:
                for ii, label in enumerate(det_res['predictions'][0]['labels']):
                    if label == 0 and det_res['predictions'][0]['scores'][ii] > 0.7:
                        xmin, ymin, xmax, ymax = det_res['predictions'][0]['bboxes'][ii]
                        bboxes.append(det_res['predictions'][0]['bboxes'][ii])
                        scores.append(det_res['predictions'][0]['scores'][ii])
                        cls_res = self.cls_inferencer(frame[int(ymin):int(ymax), int(xmin):int(xmax), :])
                        if cls_res[0]['pred_score'] > 0.6:
                            labels.append(cls_res[0]['pred_class'])
                if 0 < len(bboxes) == len(labels):
                    frame_res = draw_bbox(frame, bboxes, labels)
                    # cv2.imshow("frame", frame_res)
                    # cv2.waitKey(0)
                    # cv2.destroyWindow()
                    videoWriter.write(frame_res)
                else:
                    videoWriter.write(frame)
            except Exception as e:
                print(f"error frame {i},", e)
                videoWriter.write(frame)
        videoWriter.release()


if __name__ == "__main__":
    video_name = "/home/fangzhou/视频/xiangsheng_cut.mp4"
    peopleRec = PeopleRec()
    peopleRec.infer(video_name)
