import numpy as np
import cv2
import os

# -*- coding: utf-8 -*-
# 把几张图拼到一起
import cv2
import numpy as np


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


all_video_path1 = '/home/xyren/data/fei/save_pth/vis_bbox'
all_video_path2= '/home/xyren/data/fei/save_pth/vis_pts'

targetpath = '/home/xyren/data/fei/save_pth/video'

all_videos = os.listdir(all_video_path1)
all_videos.sort()

for video_index in range(len(all_videos)):
    video_path = all_videos[video_index]
    layer_path = os.listdir(os.path.join(all_video_path1, video_path ))
    layer_path.sort()

    targetvideopath = os.path.join(targetpath, str(video_path))
    if not os.path.exists(targetvideopath):
        os.mkdir(targetvideopath)

    num_image = len(os.listdir(os.path.join(all_video_path1, video_path, layer_path[0] )))
    for image_index in range(num_image):
        for layer_index in range(7):
            img_path = os.listdir(os.path.join(all_video_path1, video_path, layer_path[layer_index]))

            img_path.sort()
            read_img1 =os.path.join(all_video_path1, video_path, layer_path[layer_index], img_path[image_index])
            read_img2 =os.path.join(all_video_path2, video_path, layer_path[layer_index], img_path[image_index])

            print(read_img1)
            print(read_img2)
            img_1 = cv_imread(read_img1)
            img_2 = cv_imread(read_img2)

            import cv2

            cv2.putText(img_1, "Point set=2; Layer " + str(layer_index), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img_2, "Point set=3; Layer " + str(layer_index), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 1, cv2.LINE_AA)
            img_cat = np.concatenate([img_1, img_2], axis=0)

            if layer_index == 0:
                img_out = img_cat
            else:
                img_out = np.concatenate([img_out, img_cat], axis=1)  # axis=0纵向  axis=1横向
        targetvideo_imgpath = os.path.join(targetvideopath, str(image_index)+'.jpg')
        cv2.imencode('.jpg', img_out)[1].tofile(targetvideo_imgpath)


