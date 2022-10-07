import random

import matplotlib.pyplot as plt
import numpy as np
import cv2, json, os
from tqdm import tqdm

COLOR_WHITE = [255, 255, 255]
COLOR_BLACK = [0, 0, 0]
COLOR_GREY = [122, 128, 114]
COLOR_EDGE_THRESHOLD = [180, 180, 180]
DSIZE = 400


def gene_img(root_path='/home/boss/research/plantLeafDirect/', leaf_num=None, bunch_num=1, random_leaf=False):
    file_list, img_id_dict, marks_dict = standardize_src_files(os.path.join(root_path, 'dataset/coco-annotator-master/datasets/SigLeaf'))
    leaf_list = gene_leaf_list(file_list, img_id_dict, marks_dict, leaf_num=leaf_num, bunch_num=bunch_num, random_leaf=random_leaf)
    bunch, mark_list = combine_leaves(leaf_list)
    bg_bunch = add_background(bunch, bg_path=os.path.join(root_path, 'bg'))
    out_img = edge_filter(bg_bunch)
    return out_img, mark_list


def add_background(img, bg_path):
    bg_path = os.path.join(bg_path, str(random.randint(1,5)) + '.png')
    bg_img = cv2.imread(bg_path)[:DSIZE, :DSIZE]
    bg_std_img = cv2.copyMakeBorder(bg_img, 0, DSIZE-bg_img.shape[0], 0, DSIZE-bg_img.shape[1], cv2.BORDER_WRAP)
    bg_std_img = np.abs(cv2.GaussianBlur(bg_std_img, (11, 11), 2, 2) - 32)
    # plt.imshow(bg_std_img)
    # plt.show()
    # exit(0)
    content_img = np.where(img == COLOR_WHITE, bg_std_img, img)
    return content_img


def edge_filter(img, offset=20):
    img_noise = (np.sum((img >= COLOR_EDGE_THRESHOLD), axis=-1) != 3)
    img_noise = np.repeat(np.expand_dims(img_noise, axis=-1), 3, axis=-1)
    img_noise_reverse = (img_noise == False)
    clean_img = img * img_noise
    left_offset = cv2.copyMakeBorder(img[offset:, offset:, :], 0, offset, 0, offset, cv2.BORDER_CONSTANT, value=COLOR_GREY)
    fixed_img = clean_img + left_offset * img_noise_reverse
    return fixed_img


def combine_leaves(leaf_list, size=DSIZE, is_plot_bbox=False):
    center_pos = [size//2, size//2]
    new_img = np.ones([size, size, 3], dtype=np.uint8) * 255
    mark_list = []
    for leaf in leaf_list:
        mark_list.append({
            'head': [leaf['head'][0],
                     leaf['head'][1]],
            'tail': [leaf['tail'][0],
                     leaf['tail'][1]],
            'bbox_anchor_1': [leaf['bbox'][0],
                     leaf['bbox'][1]],
            'bbox_anchor_2': [leaf['bbox'][2],
                     leaf['bbox'][3]],
        })
        new_img = np.where(leaf['img'] != COLOR_WHITE, leaf['img'], new_img)

    for idx, leaf in enumerate(leaf_list):
        # coverage area ratio
        leaf_area = np.sum((leaf['img'] != COLOR_WHITE), axis=-1) == 3
        equal_mat = np.sum((leaf['img'] == new_img), axis=-1) == 3
        compare_mat = leaf_area * equal_mat
        mark_list[idx]['coverage'] = np.sum(compare_mat) / np.sum(leaf_area)
    if is_plot_bbox:
        for mark in mark_list:
            plot_bbox(mark)
    # plt.imshow(new_img)
    # plot_bbox(mark_list[-1])
    # plt.show()
    # exit(0)
    return new_img, mark_list

def plot_bbox(mark, color = [0, 255, 128]):
    plt.scatter([mark['head'][0], mark['tail'][0]], [mark['head'][1], mark['tail'][1]])
    plt.gca().add_patch(plt.Rectangle(
        (mark['bbox_anchor_1'][0], mark['bbox_anchor_1'][1]),
        mark['bbox_anchor_2'][0] - mark['bbox_anchor_1'][0],
         mark['bbox_anchor_2'][1] - mark['bbox_anchor_1'][1],
        edgecolor=[c/255 for c in color],
        fill = False, linewidth = 0.5
    ))
    return True

def gene_leaf_list(file_list, img_id_dict, marks_dict, leaf_num_center=10, leaf_num_scale=1, leaf_num=None, bunch_num=1, random_leaf=False):
    leaf_list = []
    if leaf_num is None:
        leaf_num = int(np.random.normal(loc=leaf_num_center, scale=leaf_num_scale))
    for i in range(leaf_num):
        leaf_img, leaf_bbox, leaf_head, leaf_tail = get_random_leaf(file_list, img_id_dict, marks_dict, random_leaf) # return: leaf_img, leaf_bbox, leaf_head, leaf_tail
        leaf_list.append({
            'img': leaf_img,
            'shape': leaf_img.shape,
            'bbox': leaf_bbox,
            'head': leaf_head,
            'tail': leaf_tail
        })
    return leaf_list


def standardize_src_files(path):
    whole_annotate_dict = json.load(open(os.path.join(path, '.exports', 'coco-1665029731.8240724.json')))
    img_id_dict = {item['path'].split('/')[-1]: item['id'] for item in whole_annotate_dict['images']}
    marks_dict = {anno['image_id']: {} for anno in whole_annotate_dict['annotations']}
    cate_dict = {cate['name']: cate['id'] for cate in whole_annotate_dict['categories']}
    for anno in whole_annotate_dict['annotations']:
        if anno['category_id'] == cate_dict['leafbox']:
            marks_dict[anno['image_id']]['bbox'] = anno['bbox']
        elif anno['category_id'] == cate_dict['head']:
            marks_dict[anno['image_id']]['head'] = anno['keypoints'][:2]
        elif anno['category_id'] == cate_dict['tail']:
            marks_dict[anno['image_id']]['tail'] = anno['keypoints'][:2]
        else:
            assert 'category unknown'
    file_list = [os.path.join(path, file) for file in os.listdir(path) if 'png' in file]
    return file_list, img_id_dict, marks_dict


def get_random_leaf(file_list, img_id_dict, marks_dict, random_leaf):
    img_file = random.choice(file_list)
    img_id = img_id_dict[img_file.split('/')[-1]]
    img_bbox = marks_dict[img_id]['bbox']
    img_head = marks_dict[img_id]['head']
    img_tail = marks_dict[img_id]['tail']
    img = cv2.imread(img_file)
    r_bbox, r_head, r_img, r_tail = rotate_leaf(img, tail=img_tail, head=img_head, bbox=img_bbox, random_leaf=random_leaf)
    return r_img, r_bbox, r_head, r_tail


def rotate_leaf(img, tail, head, bbox, random_leaf):

    def trans_pixs(pos, r_mat):
        return np.array([
        pos[0] * r_mat[0][0] + pos[1] * r_mat[0][1] + r_mat[0][2], # new x
        pos[0] * r_mat[1][0] + pos[1] * r_mat[1][1] + r_mat[1][2] # new y
    ])

    def trans_img(img, tail, angle, dsize=(DSIZE, DSIZE), borderValue=COLOR_WHITE):
        if random.random() <= 0 and random_leaf:
            new_center_x = random.randint(0, DSIZE)
            new_center_y = random.randint(0, DSIZE)
        else:
            new_center_x = DSIZE//2
            new_center_y = DSIZE//2
        r_mat = cv2.getRotationMatrix2D((new_center_y + (DSIZE//2), new_center_x + (DSIZE//2)), angle, 1)
        new_img = np.ones(shape=[dsize[0]*2, dsize[1]*2, 3], dtype=np.uint8) * COLOR_WHITE[0]
        # new_img[:, :, 0] = COLOR_GREY[0]
        # new_img[:, :, 1] = COLOR_GREY[1]
        # new_img[:, :, 2] = COLOR_GREY[2]
        tail_x, tail_y = tail
        img_high, img_width = img.shape[:2]
        x_start = new_center_x - tail_x + (DSIZE//2)
        x_end = new_center_x + (img_width - tail_x) + (DSIZE//2)
        y_start = new_center_y - tail_y + (DSIZE//2)
        y_end = new_center_y + (img_high - tail_y) + (DSIZE//2)
        # new_img[:img_high, :img_width] = img
        new_img[y_start:y_end, x_start:x_end] = img
        r_img = cv2.warpAffine(new_img, r_mat, dsize=(DSIZE*2, DSIZE*2), borderValue=borderValue)
        return r_img[(DSIZE//2):int(DSIZE*1.5), (DSIZE//2):int(DSIZE*1.5)], (new_center_x, new_center_y)

    def find_closest_tail_point(tail, bbox):    # input bbox, like [4, 3, 125, 84] (y, x)
        bbox_tail_distance = []
        # reallocate bbox anchor point
        bbox_tail_distance.append([abs(bbox[0] - tail[0]) + abs(bbox[1] - tail[1]), [bbox[0], bbox[1], bbox[2], bbox[3]]]) # tail pos close to bbox[0, 1]
        bbox_tail_distance.append([abs(bbox[0] - tail[0]) + abs(bbox[3] - tail[1]), [bbox[0], bbox[3], bbox[2], bbox[1]]]) # tail pos close to bbox[0, 3]
        bbox_tail_distance.append([abs(bbox[2] - tail[0]) + abs(bbox[1] - tail[1]), [bbox[2], bbox[1], bbox[0], bbox[3]]]) # tail pos close to bbox[2, 1]
        bbox_tail_distance.append([abs(bbox[2] - tail[0]) + abs(bbox[3] - tail[1]), [bbox[2], bbox[3], bbox[0], bbox[1]]]) # tail pos close to bbox[2, 3]
        bbox_tail_distance.sort()
        return bbox_tail_distance[0][1]

    def gene_rotated_max_bbox(bbox, r_mat):
        anchor_list = []
        anchor_list.append(trans_pixs(bbox[:2], r_mat))
        anchor_list.append(trans_pixs(bbox[2:], r_mat))
        anchor_list.append(trans_pixs([bbox[0], bbox[3]], r_mat))
        anchor_list.append(trans_pixs([bbox[2], bbox[1]], r_mat))
        x_list = [a[0] for a in anchor_list]
        x_list.sort()
        y_list = [a[1] for a in anchor_list]
        y_list.sort()
        return np.array([x_list[0] * 0.9, y_list[0] * 0.9, x_list[-1] * 1.1, y_list[-1] * 1.1])

    def search_img_edge(img):
        img_sum = np.sum(img, axis=-1)
        img_sum_y = np.sum(img_sum, axis=0)
        img_sum_x = np.sum(img_sum, axis=1)
        bbox = []

        for i in range(len(img_sum_y)):
            if img_sum_y[0] != img_sum_y[i]:
                bbox.append(i)
                break
        for i in range(len(img_sum_x)):
            if img_sum_x[0] != img_sum_x[i]:
                bbox.append(i)
                break

        for i in range(len(img_sum_y) - 1, 0, -1):
            if img_sum_y[0] != img_sum_y[i]:
                bbox.append(i)
                break
        for i in range(len(img_sum_x) - 1, 0, -1):
            if img_sum_x[0] != img_sum_x[i]:
                bbox.append(i)
                break

        return bbox

    random_angle = random.randint(-180, 180)
    r_mat = cv2.getRotationMatrix2D((tail[0], tail[1]), random_angle, 1)
    r_img, r_tail = trans_img(img, tail, random_angle, dsize=(DSIZE, DSIZE), borderValue=COLOR_WHITE)
    r_head = trans_pixs(head, r_mat) - np.array(tail) + np.array(r_tail)
    # t_bbox = find_closest_tail_point(tail, bbox)
    # m_bbox = gene_rotated_max_bbox(bbox, r_mat)
    # bbox_1 = m_bbox[:2] - np.array(tail) + np.array(r_tail)
    # bbox_2 = m_bbox[2:] - np.array(tail) + np.array(r_tail)

    # r_bbox = []
    # r_bbox.extend(bbox_1)
    # r_bbox.extend(bbox_2)
    r_bbox = search_img_edge(r_img)
    return r_bbox, r_head, r_img, r_tail


if __name__ == '__main__':
    img, marks = gene_img(leaf_num=16)
    for mark in marks:
        plot_bbox(mark)
    plt.imshow(img)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
