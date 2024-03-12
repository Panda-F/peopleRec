import os

import cv2


def frame_cut(video_path, target_path):
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    frame_index = 0
    if capture.isOpened():
        while True:
            ret, img = capture.read()
            if frame_index % 100 == 0:
                frame_path = os.path.join(target_path, f"frame_{frame_index}.jpg")
                cv2.imwrite(frame_path, img)
                print(f"frame cut image save to {frame_path}")
            frame_index += 1
            if not ret:
                break
    capture.release()


if __name__ == "__main__":
    video_path = "/home/fangzhou/视频/1226522947-1-192.mp4"
    target_path = "/home/fangzhou/文档/frame_cut/xiangsheng"
    frame_cut(video_path, target_path)
    # for i,frame in enumerate(video_frames):
    #     pass
