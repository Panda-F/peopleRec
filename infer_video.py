import cv2
import numpy as np
import torch
from mmdet.apis import DetInferencer
from PIL import Image


def nms(labels, scores, bboxes, iou_thresh):
    """ 非极大值抑制 """
    labels = np.array(labels)
    scores = np.array(scores)
    bboxes = np.array(bboxes)
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    keep = []

    # 按置信度进行排序
    index = np.argsort(scores)[::-1]

    while (index.size):
        # 置信度最高的框
        i = index[0]
        keep.append(index[0])

        if (index.size == 1):  # 如果只剩一个框，直接返回
            break

        # 计算交集左下角与右上角坐标
        inter_x1 = np.maximum(x1[i], x1[index[1:]])
        inter_y1 = np.maximum(y1[i], y1[index[1:]])
        inter_x2 = np.minimum(x2[i], x2[index[1:]])
        inter_y2 = np.minimum(y2[i], y2[index[1:]])
        # 计算交集的面积
        inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)
        # 计算当前框与其余框的iou
        iou = inter_area / (areas[index[1:]] + areas[i] - inter_area)
        ids = np.where(iou < iou_thresh)[0]
        index = index[ids + 1]

    return labels[keep], scores[keep], bboxes[keep]


def infer_video(video_real_path, video_virtual_path, output_name, x_offset=0, y_offset=0):
    inferencer = DetInferencer(model='/home/fangzhou/Checkpoints/xiangsheng_0311_dino/dino-4scale_r50_8xb2-12e_coco.py',
                               weights='/home/fangzhou/Checkpoints/xiangsheng_0311_dino/epoch_12.pth', device="cuda:0")

    video_real = cv2.VideoCapture(video_real_path)
    video_virtual = cv2.VideoCapture(video_virtual_path)
    fps_real = video_real.get(cv2.CAP_PROP_FPS)
    fps_virtual = video_virtual.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    size = (int(video_real.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_real.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    output = cv2.VideoWriter(f"{output_name}", fourcc, 30, size)

    real_frames = []
    virtual_frames = []
    temp = 0
    # 读取实景与虚景视频
    while video_real.isOpened():
        retval, frame_real = video_real.read()
        temp += 1
        if temp > 300:
            real_frames.append(frame_real)
        if len(real_frames) > 300:
            break
    video_real.release()
    temp = 0
    while video_virtual.isOpened():
        retval, frame_virtual = video_virtual.read()
        h, w, _ = frame_virtual.shape
        frame_virtual = cv2.resize(frame_virtual, (int(w * 0.2), int(h * 0.2)))
        frame = Image.fromarray(frame_virtual)
        frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
        temp += 1
        # if temp > 30:
        virtual_frames.append(frame[:, :, ::-1])
        # virtual_frames.append(np.resize(frame_virtual, (200, 200, 3)))
        if len(virtual_frames) == 300:
            break
    video_virtual.release()

    print("视频读取完毕")

    # 实例分割
    predictions = inferencer(real_frames)['predictions']
    for i, prediction in enumerate(predictions):
        res_img = real_frames[i]
        print(f"process frame {i}")
        thresh_index = -1
        labels = prediction['labels']
        bboxes = prediction['bboxes']
        scores = prediction['scores']
        sign_index = -1
        for ii, score in enumerate(scores):
            if labels[ii] == 1:
                sign_index = ii
                break
        x1, y1, x2, y2 = bboxes[sign_index]
        if sign_index != -1 and i < len(virtual_frames):
            try:
                res_img = merge_image(virtual_frames[i], real_frames[i], x_offset=int(x2 - 70), y_offset=int(y1 - 100))
            except Exception as e:
                print(e)
                cv2.imwrite(f"frame_real_{i}.jpg", real_frames[i])
                cv2.imwrite(f"frame_virtual_{i}.jpg", virtual_frames[i])
                # return
            # cv2.rectangle(real_frames[i], (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # cv2.imwrite(f"res_{i}.jpg", res_img)
            # return
        output.write(res_img)
    output.release()


def merge_image(front, background, x_offset=0, y_offset=0):
    front = torch.tensor(front.copy())
    front_h, front_w, _ = front.shape
    background = torch.tensor(background)
    background_h, background_w, _ = background.shape
    canvas = torch.zeros_like(background)
    # front_h - (y_offset + front_h - background_h
    # 背景融合范围
    canvas_x1 = max(0, y_offset)
    canvas_x2 = min(y_offset + front_h, background_h)
    canvas_y1 = max(0, x_offset)
    canvas_y2 = min(x_offset + front_w, background_w)
    # 前景融合范围
    front_x1 = max(0, -y_offset)
    front_x2 = min(max(background_h - y_offset, 0), front_h)
    front_y1 = max(0, -x_offset)
    front_y2 = min(max(background_w - x_offset, 0), front_w)
    canvas[canvas_x1:canvas_x2, canvas_y1:canvas_y2] = front[front_x1:front_x2, front_y1:front_y2]
    mask = torch.sum(canvas, dim=2) > 0
    background[mask > 0] = 0
    merged_img = background + canvas
    # cv2.imwrite("res_0.jpg", merged_img.numpy())
    return merged_img.numpy()


if __name__ == '__main__':
    import time

    # image_path_1 = "./frame_real_7.jpg"
    # image_path_2 = "./frame_virtual_7.jpg"
    # image_1 = cv2.imread(image_path_1)
    # image_2 = cv2.imread(image_path_2)
    # h, w, _ = image_2.shape
    # image_2 = cv2.resize(image_2, (int(w), int(h)))

    video_path_1 = "/home/fangzhou/视频/1226522947-1-192.mp4"
    video_path_2 = "/home/fangzhou/Project/RobustVideoMatting/output.mp4"
    output_path = "./res.mp4"
    start_time = time.time()
    infer_video(video_path_1, video_path_2, output_path)
    # merge_image(image_2, image_1, x_offset=0, y_offset=400)
    print(f"total time {time.time() - start_time}")
