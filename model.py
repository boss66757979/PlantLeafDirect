import time

import torch, torchvision, os, cv2
from data import ClassDataset, DSIZE
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection.rpn import AnchorGenerator
from coco_api import transforms, utils, engine, train
from coco_api.engine import train_one_epoch, evaluate
from coco_api.utils import collate_fn
import matplotlib.pyplot as plt
import numpy as np

class PlantAnalyzeModel(object):

    def __init__(self, backbone='resnet50', dataset_path=None, img_num=10, log_dir=None,
                 epochs=10, optim='sgd', lr=1e-3, batch=3, lr_gamma=0.9, leaf_num=None,
                 confidence=0.6, crowd_threshold=0.5, is_load=False, test_img_path=None):
        self.backbone = backbone
        self.dataset_path = dataset_path
        self.img_num_for_one_epoch = img_num
        self.epochs = epochs
        self.optim = optim
        self.lr = lr
        self.batch = batch
        self.lr_gamma = lr_gamma
        self.leaf_num = leaf_num
        self.is_load = is_load
        self.confidence = confidence
        self.crowd_threshold = crowd_threshold
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.save_path = './models/save/{0}_{1}_{2}.pth'.format(self.backbone, self.leaf_num, self.optim)
        self.test_img_path = test_img_path
        self.dataset_train = ClassDataset(
            self.dataset_path, img_path=self.dataset_path,
            total_len=self.img_num_for_one_epoch, leaf_num=self.leaf_num
        )
        self.log_dir = log_dir
        if self.log_dir is not None:
            self.summary_writer = SummaryWriter(log_dir=self.log_dir)

    def get_model(self, num_keypoints):
        anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512),
                                           aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
        if self.backbone == 'resnet50':
            model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
                pretrained=False,
                pretrained_backbone=True,
                num_keypoints=num_keypoints,
                num_classes=2,
                # Background is the first class, object is the second class
                rpn_anchor_generator=anchor_generator
            )

        if self.is_load:
            print('load model from: ', self.save_path)
            state_dict = torch.load(self.save_path)
            model.load_state_dict(state_dict)

        return model

    def train(self):

        data_loader_train = DataLoader(self.dataset_train, batch_size=self.batch, shuffle=True, collate_fn=collate_fn)
        # data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

        model = self.get_model(num_keypoints=2)
        model.to(self.device)

        params = [p for p in model.parameters() if p.requires_grad]
        if self.optim == 'sgd':
            optimizer = torch.optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=0.0005)
        elif self.optim == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=0.0005)
        else:
            assert 'unknown optimizer'
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=self.lr_gamma)

        min_loss = 256
        for epoch in range(self.epochs):
            metric_logger = train_one_epoch(model, optimizer, data_loader_train, self.device, epoch, print_freq=1000)
            lr_scheduler.step()
            current_loss = metric_logger.loss_keypoint.value
            if self.log_dir is not None:
                self.summary_writer.add_scalar('loss', metric_logger.loss.value, epoch)
                self.summary_writer.add_scalar('loss_keypoint', metric_logger.loss_keypoint.value, epoch)
                self.summary_writer.add_scalar('loss_box_reg', metric_logger.loss_box_reg.value, epoch)
            if current_loss <= min_loss:
                min_loss = current_loss
                evaluate(model, data_loader_train, self.device)
                # Save model weights after training
                torch.save(model.state_dict(), self.save_path)
        if self.log_dir is not None:
            self.summary_writer.close()

    def visualize(self, image, bboxes, keypoints, save_img='display.png', image_original=None, bboxes_original=None, keypoints_original=None):
            keypoints_classes_ids2names = {
                0: 'h', 1: 't'}

            for bbox in bboxes:
                start_point = (bbox[0], bbox[1])
                end_point = (bbox[2], bbox[3])
                image = cv2.rectangle(image.copy(), start_point, end_point, (0, 255, 0), 2)

            for kps in keypoints:
                for idx, kp in enumerate(kps):
                    image = cv2.circle(image.copy(), tuple(kp), 1, (255, 0, 0), 1)
                    image = cv2.putText(image.copy(), " " + keypoints_classes_ids2names[idx], tuple(kp),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 128, 128), 1, cv2.LINE_AA)

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
                fontsize = 2

                ax[0].imshow(image_original)
                ax[0].set_title('Original image', fontsize=fontsize)

                ax[1].imshow(image)
                ax[1].set_title('Transformed image', fontsize=fontsize)

            plt.savefig(os.path.join(self.dataset_path, 'result/img', save_img))

    def display(self):

        data_loader = DataLoader(self.dataset_train, batch_size=1, shuffle=True, collate_fn=collate_fn)

        iterator = iter(data_loader)
        batch = next(iterator)

        # print("Original targets:\n", batch[3], "\n\n")
        print("Transformed targets:\n", batch[1])


        image = (batch[0][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        bboxes = batch[1][0]['boxes'].detach().cpu().numpy().astype(np.int32).tolist()

        keypoints = []
        for kps in batch[1][0]['keypoints'].detach().cpu().numpy().astype(np.int32).tolist():
            keypoints.append([kp[:2] for kp in kps])

        self.visualize(image, bboxes, keypoints)

    def predict(self, use_images=False):
        model = self.get_model(num_keypoints=2)
        images = []
        if self.test_img_path is not None and use_images:
            for img_file in os.listdir(self.test_img_path):
                img = cv2.imread(os.path.join(self.test_img_path, img_file))
                img = cv2.copyMakeBorder(img, 0, DSIZE-img.shape[0], 0, DSIZE-img.shape[1], cv2.BORDER_CONSTANT, value=[0, 0, 0])
                images.append(img.transpose(2, 0, 1))
            images = np.array(images, dtype=np.float32) / 255.0
            with torch.no_grad():
                model.to(self.device)
                images = torch.tensor(images).to(self.device)
                model.eval()
                output = model(images)
            for i in range(len(images)):
                image = (images[i].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                scores = output[i]['scores'].detach().cpu().numpy()

                high_scores_idxs = np.where(scores > self.confidence)[0].tolist()  # Indexes of boxes with scores > 0.7
                post_nms_idxs = torchvision.ops.nms(output[i]['boxes'][high_scores_idxs],
                                                    output[i]['scores'][high_scores_idxs],
                                                    0.3).cpu().numpy()  # Indexes of boxes left after applying NMS (iou_threshold=0.3)

                keypoints = []
                for kps in output[i]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                    keypoints.append([list(map(int, kp[:2])) for kp in kps])

                bboxes = []
                # for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                #     bboxes.append(list(map(int, bbox.tolist())))

                self.visualize(image, bboxes, keypoints, save_img='predict_{0}.png'.format(i))
        else:
            data_loader = DataLoader(self.dataset_train, batch_size=1, shuffle=True, collate_fn=collate_fn)

            iterator = iter(data_loader)
            batch = next(iterator)

            with torch.no_grad():
                model.to(self.device)
                images = batch[0][0]
                images = images.unsqueeze(0).to(self.device)
                model.eval()
                output = model(images)

            print("Predictions: \n", output)
            image = (batch[0][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            # image = (images[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            scores = output[0]['scores'].detach().cpu().numpy()

            high_scores_idxs = np.where(scores > self.confidence)[0].tolist()  # Indexes of boxes with scores > 0.7
            post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs],
                                                0.3).cpu().numpy()  # Indexes of boxes left after applying NMS (iou_threshold=0.3)

            # Below, in output[0]['keypoints'][high_scores_idxs][post_nms_idxs] and output[0]['boxes'][high_scores_idxs][post_nms_idxs]
            # Firstly, we choose only those objects, which have score above predefined threshold. This is done with choosing elements with [high_scores_idxs] indexes
            # Secondly, we choose only those objects, which are left after NMS is applied. This is done with choosing elements with [post_nms_idxs] indexes

            keypoints = []
            for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                keypoints.append([list(map(int, kp[:2])) for kp in kps])

            bboxes = []
            # for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
            #     bboxes.append(list(map(int, bbox.tolist())))

            self.visualize(image, bboxes, keypoints, save_img='predict.png')