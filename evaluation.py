# import cv2
import numpy as np
import os
import cv2

def iou(image1, image2):
    intersection = np.logical_and(image1, image2)
    union = np.logical_or(image1, image2)
    iou_score = np.sum(intersection) / np.sum(union)
    if np.isnan(iou_score):
        iou_score = 0
    return iou_score


def main(predict_folder, groundtruth_folder, metric):
    iou_scores = {}
    print(f"Calculating {metric} scores...")
    count = 0
    sum = 0

    for filename in os.listdir(predict_folder):
        predict_path = os.path.join(predict_folder, filename)
        groundtruth_file_name = os.path.splitext(filename)[0] + '.png'
        groundtruth_path = os.path.join(groundtruth_folder, groundtruth_file_name)

        if os.path.exists(groundtruth_path):
            predict_image = cv2.imread(predict_path, cv2.IMREAD_GRAYSCALE)
            groundtruth_image = cv2.imread(groundtruth_path, cv2.IMREAD_GRAYSCALE)

            if metric == 'iou':
                iou_score = iou(predict_image, groundtruth_image)
                # print(iou_score)
                iou_scores[filename] = iou_score
                count += 1
                sum += iou_score

        else:
            print(f"Groundtruth file not found for {filename}")
    print(f"average {metric} score:  {sum / count} ")
    return iou_scores, sum / count


predict_folder = '../masks_hsv'
groundtruth_folder = '../masks_test'

# iou_scoresa, hsv = main('./masks_hsv', groundtruth_folder, 'iou')
# iou_scoresb, ori = main('./masks_ori', groundtruth_folder, 'iou')
# iou_scoresc, lumin = main('./masks_lumin', groundtruth_folder, 'iou')
# iou_scoresc, fdog = main('./masks_fdog', groundtruth_folder, 'iou')

# Calculating iou scores...
hsv = 0.6807698367099251 
# Calculating iou scores...
ori = 0.7527329226062384 
# Calculating iou scores...
lumin = 0.7733338977518011 
# Calculating iou scores...
fdog = 0.7984794726003903
import pandas as pd
data = pd.DataFrame([[hsv, ori, lumin, fdog]])
print(data)
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(x=[1,2,3,4], y=data, marker='o', c='r')
plt.plot([1,2,3,4], [hsv, ori, lumin, fdog], marker='o')
plt.xticks([1,2,3,4], ['hsv', 'ori', 'lumin', 'fdog'],rotation=45)
plt.ylim((0.5, 1))
plt.ylabel("test iou")
plt.show()