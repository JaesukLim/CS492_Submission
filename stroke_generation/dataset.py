## This code is modified from https://github.com/KAIST-Visual-AI-Group/Diffusion-Assignment2-DDIM-CFG/blob/main/image_diffusion_todo/dataset.py
## and https://github.com/KAIST-Visual-AI-Group/Diffusion-Project-Drawing/blob/master/load_data.ipynb

import json
import ndjson
import numpy as np
from itertools import chain
from pathlib import Path
import torch

def listdir(dname):
    fnames = list(
        chain(
            *[
                list(Path(dname).rglob("*." + ext))
                for ext in ["png", "jpg", "jpeg", "JPG"]
            ]
        )
    )
    return fnames

def image_grid(imgs, rows, cols):
    """
    Concatenates multiple images
    """
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def draw_strokes(strokes, height=256, width=256):
    """
    Make a new PIL image with the given strokes
    """
    image = Image.new("RGB", (width, height), "white")
    image_draw = ImageDraw.Draw(image)
    for stroke in strokes:
        points = [(int(x), int(y)) for x, y in stroke]
        image_draw.line(points, fill=0)
    return image

from PIL import Image, ImageDraw
def tensor_to_pil_image(x: torch.Tensor, single_image=False):
    """
    x: [B,C,H,W]
    """
    output = []
    for k in range(x.shape[0]):
        min_val = x[k][0:2].min()
        max_val = x[k][0:2].max()
        x[k][0:2] = (x[k][0:2] - min_val) / (max_val - min_val) * 255
        strokes = x[k].squeeze()

        input_strokes = []
        current_stroke = []  # 현재 그룹을 저장할 임시 리스트

        for i in range(len(strokes[0])):
            # 현재 좌표 계산
            current_coords = [strokes[0][i], strokes[1][i]]
            
            if strokes[2][i] <= 0:
                current_stroke.append(current_coords)
            else:
                # 조건을 만족하지 않을 경우, 현재 그룹을 input_strokes에 추가
                if current_stroke:  # 그룹이 비어 있지 않으면 추가
                    current_stroke.append(current_coords)
                    input_strokes.append(current_stroke)
                    current_stroke = []  # 그룹 초기화

        # 루프 종료 후, 남은 그룹 추가
        if current_stroke:
            input_strokes.append(current_stroke)

        images = []
        for i in range(len(input_strokes)):
            image = draw_strokes(input_strokes[:i+1])

            # add stroke number
            draw = ImageDraw.Draw(image)
            draw.text((20, 10), text=f"stroke #{i}", fill="black")
            images.append(image)

        # concatenate all images
        images_concat = image_grid(images, 1, len(images))
        output.append(images_concat)
    return output


def get_data_iterator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
    for i, data in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


class QuickDrawDataset(torch.utils.data.Dataset):
    def __init__(
        self, root: str, indices_root: str, split: str, category="cat", label_offset=1, num_classes=1, coord_length=150
    ):
        super().__init__()
        self.root = root
        self.indices_root = indices_root
        self.split = split
        self.label_offset = label_offset
        self.category = category
        self.num_classes = num_classes
        self.coord_length = coord_length

        data_path = f"{root}/{category}.ndjson"
        indices_path = f"{indices_root}/{category}/train_test_indices.json"

        with open(data_path, 'r') as f:
            self.data = ndjson.load(f)

        with open(indices_path, 'r') as f:
            indices = json.load(f)
        self.indices = indices[split]
        self.labels = [label_offset] * len(self.indices)  # 모든 파일의 라벨

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        stroke_data = self.data[actual_idx]
        label = self.labels[idx]

        coords = []
        stroke_data_out = []
        
        for stroke in stroke_data['drawing']:
            coordinate_length = len(stroke[0])
            new_stroke = np.zeros((3, coordinate_length))
            new_stroke[:2, :] = stroke
            new_stroke[2, -1] = 1  # 마지막 coordinate에만 1
            stroke_data_out.append(new_stroke)
        stroke_data_out = np.concatenate(stroke_data_out, axis=1)

        index_count = 1
        # 길이가 96보다 작을 경우 중간값 추가
        while stroke_data_out.shape[1] < self.coord_length:
            if index_count < stroke_data_out.shape[1] and stroke_data_out[2][index_count-1] != 1:
                mid_x = (stroke_data_out[0][index_count - 1] + stroke_data_out[0][index_count]) / 2
                mid_y = (stroke_data_out[1][index_count - 1] + stroke_data_out[1][index_count]) / 2
                # 좌표 삽입
                stroke_data_out = np.insert(stroke_data_out, index_count, [mid_x, mid_y, 0], axis=1)
                index_count += 2
            if stroke_data_out[2, index_count-1] == 1:
                index_count += 1
            if index_count >= stroke_data_out.shape[1]:
                index_count = 1

        # 길이 넘어가면 Clip
        if stroke_data_out.shape[1] > self.coord_length:
            stroke_data_out = stroke_data_out[:, :self.coord_length]
        stroke_data_out = torch.tensor(stroke_data_out).unsqueeze(dim=1).float()
        stroke_data_out[0:2] = (stroke_data_out[0:2] / 255 - 0.5) * 5
        stroke_data_out[2]=(stroke_data_out[2] - 0.5) * 5
        return stroke_data_out, label

    def __len__(self):
        return len(self.labels)


class QuickDrawDataModule(object):
    def __init__(
        self,
        root: str = "./data",
        indices_root: str = "./sketch_data",
        batch_size: int = 32,
        num_workers: int = 4,
        image_resolution: int = 256,
        label_offset=1,
        category="cat",
        coord_length = 150
    ):
        self.root = root
        self.indices_root = indices_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_resolution = image_resolution
        self.label_offset = label_offset
        self.num_classes = 1
        self.category = category
        self.coord_length = coord_length
        self._set_dataset()

        print(f"""#### Dataset Module Information ####
        
Category: {category}
Coord_length: {coord_length}
Root: {root}
Batch size: {batch_size}

####################################
""")

    def _set_dataset(self):
        self.train_ds = QuickDrawDataset(
            self.root,
            self.indices_root,
            "train",
            category=self.category,
            label_offset=self.label_offset,
            num_classes = self.num_classes,
            coord_length=self.coord_length
        )
        self.val_ds = QuickDrawDataset(
            self.root,
            self.indices_root,
            "test",
            category=self.category,
            label_offset=self.label_offset,
            num_classes = self.num_classes,
            coord_length=self.coord_length
        )


    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )

