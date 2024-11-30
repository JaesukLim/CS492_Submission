## This code is modified from https://github.com/KAIST-Visual-AI-Group/Diffusion-Project-Drawing/blob/master/load_data.ipynb

import os
import ndjson
import json
from tqdm import tqdm
from PIL import Image, ImageDraw

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
        # concat x and y coordinates
        points = list(zip(stroke[0], stroke[1]))

        # draw all points
        image_draw.line(points, fill=0)

    return image

def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

for category in ["cat", "garden", "helicopter"]:
    data_path = f"./data/{category}.ndjson"
    indices_path = f"./sketch_data/{category}/train_test_indices.json"

    with open(data_path, 'r') as f:
        data = ndjson.load(f)

    with open(indices_path, 'r') as f:
        indices = json.load(f)

    for j, idx in tqdm(enumerate(indices["test"])):
        item = data[idx]
        strokes = item['drawing']
        base_path = f"./test/{category}"
        ensure_dir_exists(base_path)
        images = []
        for i in range(len(strokes)):
            image = draw_strokes(strokes[:i + 1])
            draw = ImageDraw.Draw(image)

            draw.text((20, 10), text=f"stroke #{i}", fill="black")
            images.append(image)
        images_concat = image_grid(images, 1, len(images))
        images_concat.save(f"./test/{category}/{category}_{j}_stroke.png")


