import os
from pathlib import Path
import json

import requests
from sklearn.model_selection import train_test_split
from yaml import dump

import math

def get_rotated_rectangle(x, y, w, h, theta, original_width, original_height, out_norm=True):
    x1 = x / 100
    y1 = y / 100

    w = w * original_width
    h = h * original_height

    x2 = (x * original_width + w * math.cos(math.radians(theta))) / original_width / 100
    y2 = (y * original_height + w * math.sin(math.radians(theta))) / original_height / 100
    x3 = (x * original_width + w * math.cos(math.radians(theta)) - h * math.sin(
        math.radians(theta))) / original_width / 100
    y3 = (y * original_height + w * math.sin(math.radians(theta)) + h * math.cos(
        math.radians(theta))) / original_height / 100

    x4 = (x * original_width - h * math.sin(math.radians(theta))) / original_width / 100
    y4 = (y * original_height + h * math.cos(math.radians(theta))) / original_height / 100

    if not out_norm:
        x1 *= original_width
        x2 *= original_width
        x3 *= original_width
        x4 *= original_width

        y1 *= original_height
        y2 *= original_height
        y3 *= original_height
        y4 *= original_height

    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

def main():
    with open(data_path) as f:
        data = json.load(f)

    root_addr = f'http://{host}:{port}' 
    data_dir = data_path[:-5]
    img_dir = f'{data_dir}/images'
    label_dir = f'{data_dir}/labels'
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    with open(f'{data_dir}/classes.txt', 'w') as f:
        f.writelines(classes)
        
    class2id = {v: k for k, v in enumerate(classes)}
    for sample in data:
        sample_name = sample['file_upload']
        img_ext = Path(sample_name).suffix
        url = sample['data']['image']
        url = f'{root_addr}{url}'
        
        img_data = requests.get(url, headers={'Authorization': f'Token {token}'}).content
        with open(f'{img_dir}/{sample_name}', 'wb') as handler:
            handler.write(img_data)

        annotation = sample['annotations'][0]['result']
        label_path = f'{label_dir}/{sample_name[:-len(img_ext)]}.txt'
        with open(label_path, 'w') as f:
            for label in annotation:
                try:
                    class_name = label['value']['rectanglelabels'][0]
                except KeyError:
                    continue
                if class_name in classes:
                    img_width, img_height = label['original_width'], label['original_height']
                    x, y = label['value']['x'], label['value']['y']
                    width, height = label['value']['width'], label['value']['height']
                    rotation = label['value']['rotation']
                    [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] = get_rotated_rectangle(x, y, width, height, rotation, img_width, img_height)
                    class_id = class2id[class_name]
                    f.write(f'{class_id} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n')
                    
    img_paths = [f'{str(img_path)}\n' for img_path in Path(img_dir).glob('*.jpg')]
    train_file = 'train.txt'
    val_file = 'val.txt'
    trains, vals = train_test_split(img_paths, test_size=val_ratio)
    with open(f'{data_dir}/{train_file}', 'w') as f:
        f.writelines(trains)

    with open(f'{data_dir}/{val_file}', 'w') as f:
        f.writelines(vals)

    dataset_conf = {
        'path': str(Path(data_dir).resolve()),
        'train': train_file,
        'val': val_file,
        'names': classes
    }
    with open(f'{data_dir}/conf.yaml', 'w') as f:
        dump(dataset_conf, f)


if __name__ == '__main__':
    data_path = 'data/project-51-at-2024-07-24-16-24-a0ad0596.json'
    host = '115.77.105.238'
    port = 9511
    val_ratio = 0.2
    token = '03943361fe45461abcec0c1ac75fe70acd74b6ac'
    classes = ['stop', 'right', 'left', 'straight']
    main()
