import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_res(masks, scores, input_point, input_label, input_box, image):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            box = input_box[i]
            show_box(box, plt.gca())
        if (input_point is not None) and (input_label is not None):
            show_points(input_point, input_label, plt.gca())

        print(f"Score: {score:.3f}")
        plt.axis('off')
        plt.show()

def show_res_multi(masks, scores, input_point, input_label, input_box, image):
    plt.figure(figsize=(6,6))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    for box in input_box:
        show_box(box, plt.gca())
    for score in scores:
        print(f"Score: {score:.3f}")
    plt.axis('off')
    plt.show()

def getpos(img):
    img = img.copy()
    points = []
    labels = []
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            points.append([x,y])
            labels.append(1)
            cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
            cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (255, 255, 255), thickness=1)
            cv2.imshow("image", img)
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            points.append([x,y])
            labels.append(0)
            cv2.circle(img, (x, y), 1, (255, 255, 0), thickness=-1)
            cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (255, 255, 255), thickness=1)
            cv2.imshow("image", img)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    print(points)
    print(labels)
    return np.array(points), np.array(labels)

def getbox(img):
    img = img.copy()
    temp = []
    boxs = []
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param: list):
        global temp
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d,start" % (x, y)
            temp = [x,y]
            cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
            cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (255, 255, 255), thickness=1)
            cv2.imshow("image", img)
        elif event == cv2.EVENT_LBUTTONUP:
            xy = "%d,%d,end" % (x, y)
            if temp[0] != x or temp[1] != y:
                print(temp, [x,y])
                boxs.append(temp + [x,y])
            cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
            cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (255, 255, 255), thickness=1)
            cv2.imshow("image", img)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    print(boxs)
    return boxs

# def getmark(img):
#     temp = []
#     boxs = []
#     points = []
#     labels = []
#     def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
#         global temp
#         if event == cv2.EVENT_LBUTTONDOWN:
#             xy = "%d,%d" % (x, y)
#             temp = [x,y]
            
#             cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
#             cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
#                         1.0, (255, 255, 255), thickness=1)
#             cv2.imshow("image", img)
            
#         elif event == cv2.EVENT_LBUTTONUP:
#             if temp[0] != x or temp[1] != y:
#                 print(temp, [x,y])
#                 boxs.append(temp + [x,y])
#                 xy = "%d,%d,end" % (x, y)
#             else:
#                 points.append([x,y])
#                 labels.append(1)
#                 xy = "%d,%d" % (x, y)
#             cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
#             cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
#                         1.0, (255, 255, 255), thickness=1)
#             cv2.imshow("image", img)

#         elif event == cv2.EVENT_RBUTTONDOWN:
#             xy = "%d,%d" % (x, y)
#             points.append([x,y])
#             labels.append(0)
#             cv2.circle(img, (x, y), 1, (255, 255, 0), thickness=-1)
#             cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
#                         1.0, (255, 255, 255), thickness=1)
#             cv2.imshow("image", img)

#     cv2.namedWindow("image")
#     cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
#     cv2.imshow("image", img)
#     cv2.waitKey(0)
#     return boxs if len(boxs) != 0 else None, \
#         np.array(points) if len(points) != 0 else None, \
#         np.array(labels) if len(labels) != 0 else None