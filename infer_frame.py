from mmdet.apis import DetInferencer
import time

# 初始化模型
inferencer = DetInferencer(model='/home/fangzhou/Checkpoints/xiangsheng_0311_dino/dino-4scale_r50_8xb2-12e_coco.py', weights='/home/fangzhou/Checkpoints/xiangsheng_0311_dino/epoch_12.pth')

# 推理示例图片
start_time = time.time()
inferencer('/home/fangzhou/Dataset/xiangsheng/frame_5700.jpg')
print(f"time is {time.time()-start_time}")