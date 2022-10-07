import matplotlib.pyplot as plt
import re


def plot_loss(file_path=None, img_name=None, is_show=False):
    loss_pattern = '.*?0/12.*?loss:.*?\((.*?)\).*?loss_classifier'
    sgd_loss_list = []
    adam_loss_list = []
    for line in open(file_path + '_sgd.log', encoding='utf8').readlines():
        result = re.search(loss_pattern, line)
        if result is not None:
            sgd_loss_list.append(float(result.groups()[0]))
    for line in open(file_path + '_adam.log', encoding='utf8').readlines():
        result = re.search(loss_pattern, line)
        if result is not None:
            adam_loss_list.append(float(result.groups()[0]))
    plt.plot([n for n in range(len(sgd_loss_list))], sgd_loss_list, label='SGD training loss')
    plt.plot([n for n in range(len(adam_loss_list))], adam_loss_list, label='Adam training loss')
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if is_show:
        plt.show()
    else:
        plt.savefig(img_name)

if __name__ == '__main__':
    plot_loss(file_path='log/resnet50_16', img_name='img/resnet50_16.png')