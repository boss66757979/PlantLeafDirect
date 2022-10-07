import os, json, cv2, torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import numpy as np
from dataset.fake_gene import *


class ClassDataset(Dataset):
    def __init__(self, root, load_img=False, img_path=None, total_len=512,
                 leaf_num=None, crowd_threshold=0.5):
        self.root = root
        self.load_img = load_img
        self.img_path = img_path
        self.total_len = total_len
        self.leaf_num = leaf_num
        self.crowd_threshold = crowd_threshold
        if load_img:
            self.imgs_files = sorted(os.listdir(os.path.join(root, "test")))
            self.annotations_files = sorted(os.listdir(os.path.join(root, "annotations")))

    def __getitem__(self, idx):

        if self.load_img:
            img_path = os.path.join(self.root, "images", self.imgs_files[idx])
            annotations_path = os.path.join(self.root, "annotations", self.annotations_files[idx])

            img_original = cv2.imread(img_path)
            img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

            with open(annotations_path) as f:
                data = json.load(f)
                bboxes_original = data['bboxes']
                keypoints_original = data['keypoints']

            img, bboxes, keypoints = img_original, bboxes_original, keypoints_original

                # Convert everything into a torch tensor
            bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
            target = {}
            target["boxes"] = bboxes
            target["labels"] = torch.as_tensor([1 for _ in bboxes], dtype=torch.int64)  # all objects are glue tubes
            target["image_id"] = torch.tensor([idx])
            target["area"] = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
            target["iscrowd"] = torch.zeros(len(bboxes), dtype=torch.int64)
            target["keypoints"] = torch.as_tensor(keypoints, dtype=torch.float32)
            img = F.to_tensor(img)

            return img, target
        else:
            img, annotation = gene_img(self.img_path, leaf_num=self.leaf_num)
            boxes = [[mark['bbox_anchor_1'][0], mark['bbox_anchor_1'][1], mark['bbox_anchor_2'][0], mark['bbox_anchor_2'][1]] for mark in annotation]
            keypoints = [[[mark['head'][0], mark['head'][1], 1], [mark['tail'][0], mark['tail'][1], 0.5]] for mark in annotation]
            is_crowd = torch.as_tensor([
                (0 if mark['coverage'] >= self.crowd_threshold else 1)
                for mark in annotation], dtype=torch.int64)
            keypoints[-1][1][2] = 1
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            target = {}
            target["boxes"] = boxes
            target["labels"] = torch.as_tensor([1 for _ in boxes], dtype=torch.int64)  # all objects are glue tubes
            target["image_id"] = torch.tensor([idx])
            target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            target["iscrowd"] = is_crowd
            target["keypoints"] = torch.as_tensor(keypoints, dtype=torch.float32)
            img = F.to_tensor(img)

            return img, target

    def __len__(self):
        if self.load_img:
            return len(self.imgs_files)
        else:
            return self.total_len
