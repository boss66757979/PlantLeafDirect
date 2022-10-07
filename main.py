import argparse
from model import PlantAnalyzeModel
parser = argparse.ArgumentParser()

# model parameter
parser.add_argument("--model", default='resnet50', help='backbone model type', type=str)
parser.add_argument("--is_pred", default=True, help='predict image using trained model', type=bool)
parser.add_argument("--is_load", default=True, help='load model from save file', type=bool)
parser.add_argument("--is_display", default=False, help='display sample file', type=bool)
parser.add_argument("--log_dir", default='/home/boss/research/plantLeafDirect/result/log', help='display sample file', type=bool)
parser.add_argument("--leaf_num", default=16, help='leaf number of each image, the random number should be None', type=int)
parser.add_argument("--confidence", default=0.2, help='leaf number of each image, the random number should be None', type=int)
parser.add_argument("--crowd_threshold", default=0.3, help='leaf number of each image, the random number should be None', type=int)
parser.add_argument("--file_path", default='/home/boss/research/plantLeafDirect/', help='leaf image file', type=str)
parser.add_argument("--test_img_path", default='/home/boss/research/plantLeafDirect/dataset/test', help='leaf image file', type=str)
parser.add_argument("--img_num", default=36, help='image number for one epoch', type=int)
parser.add_argument("--batch", default=3, help='image number for one epoch', type=int)
parser.add_argument("--epochs", default=256, help='training epochs', type=int)
parser.add_argument("--optimizer", default='sgd', help='training optimizer', type=str)
parser.add_argument("--lr", default=1e-3, help='training learning rate, default is 1e-3', type=int)
parser.add_argument("--lr_gamma", default=0.9, help='learning rate decay param', type=int)


# load params
args = parser.parse_args()
print(args)
plant_analyze_model = PlantAnalyzeModel(
    backbone=args.model,
    is_load=args.is_load,
    dataset_path=args.file_path,
    img_num=args.img_num,
    epochs=args.epochs,
    optim=args.optimizer,
    lr=args.lr,
    log_dir=args.log_dir,
    batch=args.batch,
    test_img_path=args.test_img_path,
    confidence=args.confidence,
    crowd_threshold=args.crowd_threshold,
    lr_gamma=args.lr_gamma,
    leaf_num=args.leaf_num
)
if args.is_pred:
    plant_analyze_model.predict(use_images=True)
elif args.is_display:
    plant_analyze_model.display()
else:
    plant_analyze_model.train()