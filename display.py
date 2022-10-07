import numpy as np
from torch.utils.data import DataLoader
from data import ClassDataset
import cv2
import matplotlib.pyplot as plt
from coco_api.utils import collate_fn


KEYPOINTS_FOLDER_TRAIN = '/path/to/dataset/train'
dataset = ClassDataset(KEYPOINTS_FOLDER_TRAIN, demo=True)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

iterator = iter(data_loader)
batch = next(iterator)

print("Original targets:\n", batch[3], "\n\n")
print("Transformed targets:\n", batch[1])


# Original targets:
# ({'boxes': tensor([[296., 116., 436., 448.],
#                    [577., 589., 925., 751.]]),
# 'labels': tensor([1, 1]),
# 'image_id': tensor([15]),
# 'area': tensor([46480., 56376.]),
# 'iscrowd': tensor([0, 0]),
# 'keypoints': tensor([[[408., 407.,   1.],
#                       [332., 138.,   1.]],
#                      [[886., 616.,   1.],
#                       [600., 708.,   1.]]])},
# )
# Transformed targets:
# ({'boxes': tensor([[ 116., 1484.,  448., 1624.],
#                    [ 589.,  995.,  751., 1343.]]),
#   'labels': tensor([1, 1]),
#   'image_id': tensor([15]),
#   'area': tensor([46480., 56376.]),
#   'iscrowd': tensor([0, 0]),
#   'keypoints': tensor([[[4.0700e+02, 1.5110e+03, 1.0000e+00],
#                         [1.3800e+02, 1.5870e+03, 1.0000e+00]],
#                        [[6.1600e+02, 1.0330e+03, 1.0000e+00],
#                         [7.0800e+02, 1.3190e+03, 1.0000e+00]]])},
# )

keypoints_classes_ids2names = {
    0: 'Head', 1: 'Tail'}


def visualize(image, bboxes, keypoints, image_original=None, bboxes_original=None, keypoints_original=None):
    fontsize = 18

    for bbox in bboxes:
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        image = cv2.rectangle(image.copy(), start_point, end_point, (0, 255, 0), 2)

    for kps in keypoints:
        for idx, kp in enumerate(kps):
            image = cv2.circle(image.copy(), tuple(kp), 5, (255, 0, 0), 10)
            image = cv2.putText(image.copy(), " " + keypoints_classes_ids2names[idx], tuple(kp),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)

    if image_original is None and keypoints_original is None:
        plt.figure(figsize=(40, 40))
        plt.imshow(image)

    else:
        for bbox in bboxes_original:
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[2], bbox[3])
            image_original = cv2.rectangle(image_original.copy(), start_point, end_point, (0, 255, 0), 2)

        for kps in keypoints_original:
            for idx, kp in enumerate(kps):
                image_original = cv2.circle(image_original, tuple(kp), 5, (255, 0, 0), 10)
                image_original = cv2.putText(image_original, " " + keypoints_classes_ids2names[idx], tuple(kp),
                                             cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)

        f, ax = plt.subplots(1, 2, figsize=(40, 20))

        ax[0].imshow(image_original)
        ax[0].set_title('Original image', fontsize=fontsize)

        ax[1].imshow(image)
        ax[1].set_title('Transformed image', fontsize=fontsize)


image = (batch[0][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
bboxes = batch[1][0]['boxes'].detach().cpu().numpy().astype(np.int32).tolist()

keypoints = []
for kps in batch[1][0]['keypoints'].detach().cpu().numpy().astype(np.int32).tolist():
    keypoints.append([kp[:2] for kp in kps])

image_original = (batch[2][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
bboxes_original = batch[3][0]['boxes'].detach().cpu().numpy().astype(np.int32).tolist()

keypoints_original = []
for kps in batch[3][0]['keypoints'].detach().cpu().numpy().astype(np.int32).tolist():
    keypoints_original.append([kp[:2] for kp in kps])

visualize(image, bboxes, keypoints, image_original, bboxes_original, keypoints_original)