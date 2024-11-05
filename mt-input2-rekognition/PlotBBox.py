import json

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def draw_image(image_path, data, formatter):
    image = Image.open(image_path)
    annotations = data["annotations"]
    draw = ImageDraw.Draw(image)
    sample = data["images"][0]
    width, height = sample["width"], sample["height"]

    print(width, height)

    for i in range(len(annotations)):
        box = annotations[i]["bbox"]
        print(box)

        x1, y1, x2, y2 = formatter(box, width, height)

        draw.rectangle((x1, y1, x2, y2), outline="blue", width=3)
        print(x1, y1, x2, y2)
    return image

def coco_format(box, width, height):
    x, y, w, h = tuple(box)

    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    return x1, y1, x2, y2

def top_left_format(box, width, height):
    y, x, w, h = tuple(box)

    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    return x1, y1, x2, y2

def top_left_format_origin_bottom(box, width, height):
    y, x, w, h = tuple(box)

    x1, y1 = int(x), int(height - y)
    x2, y2 = int(x + w), int(height - y + h)
    return x1, y1, x2, y2

def y_at_bottom_left_format(box, width, height):
    x, y, w, h = tuple(box)

    x1, y1 = int(x), int(height - y - h)
    x2, y2 = int(x + w), int(height - y)
    return x1, y1, x2, y2

def two_points_start_top_left(box, width, height):
    x, y, x_other, y_other = tuple(box)

    x1, y1 = int(x), int(y)
    x2, y2 = int(x_other ), int(y_other)
    if x1 < x2:
        if y1 < y2:
            return x1, y1, x2, y2
        else:
            return x1, y2, x2, y1
    else:
        if y2 < y1:
            return x2, y2, x1, y1
        else:
            return x2, y1, x1, y2

def two_points_start_bottom_left(box, width, height):
    x, y, x_other, y_other = tuple(box)

    x1, y1 = int(x), int(height) - int( y)
    x2, y2 = int(x_other ), int(height) - int(y_other)
    if x1 < x2:
        if y1 < y2:
            return x1, y1, x2, y2
        else:
            return x1, y2, x2, y1
    else:
        if y2 < y1:
            return x2, y2, x1, y1
        else:
            return x2, y1, x1, y2

with open('test4.json', 'r') as f:
    cocoLike = json.load(f)

#/Users/steene/PycharmProjects/RekognitionExperiment/mt-input2/2023_uge22_bauhaus_1.jpg
#/Users/steene/PycharmProjects/RekognitionExperiment/mt-input2-rekognition/2023_uge22_bauhaus_1.jpg
# print(cocoLike["images"])
# print(cocoLike["images"][0]["file_name"])
cocoImage = draw_image("/Users/steene/PycharmProjects/RekognitionExperiment/mt-input2/"+cocoLike["images"][0]["file_name"], cocoLike, coco_format)

fig = plt.figure(figsize=(1207/100,1489/100))
fig.figimage(cocoImage)
plt.show()