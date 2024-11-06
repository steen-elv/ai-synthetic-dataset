import json

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def draw_image(image_path, data, formatter):
    image = Image.open(image_path)
    print(image.size)
    annotations = data["CustomLabels"]
    draw = ImageDraw.Draw(image)

    width, height = image.size

    print(width, height)

    for i in range(len(annotations)):
        box = annotations[i]["Geometry"]["BoundingBox"]
        print(box)

        x1, y1, x2, y2 = formatter(box, width, height)

        draw.rectangle((x1, y1, x2, y2), outline="blue", width=3)
        print(x1, y1, x2, y2)
    return image

def coco_format(box, width, height):
    print(tuple(box))
    x, y, w, h = box["Left"], box["Top"], box["Width"], box["Height"]
    print(x, y, w, h)

    x1 = int(x * width)
    y1 = int(y * height)
    x2 = int((x + w) * width)
    y2 = int((y + h) * height)
    return x1, y1, x2, y2

with open('2024_uge13_rema1000_babyogboern_1_inference.json', 'r') as f:
    customLabelRes = json.load(f)

#/Users/steene/PycharmProjects/RekognitionExperiment/mt-input2/2023_uge22_bauhaus_1.jpg
#/Users/steene/PycharmProjects/RekognitionExperiment/mt-input2-rekognition/2023_uge22_bauhaus_1.jpg
# print(cocoLike["images"])
# print(cocoLike["images"][0]["file_name"])
# cocoImage = draw_image("/Users/steene/PycharmProjects/RekognitionExperiment/mt-input2/"+cocoLike["images"][0]["file_name"], cocoLike, coco_format)
# cocoImage = draw_image("/Users/steene/PycharmProjects/RekognitionExperiment/synthetic/ad_dataset/images/ad_000001.jpg", customLabelRes, coco_format)
cocoImage = draw_image("/Users/steene/PycharmProjects/RekognitionExperiment/mt-input/2024_uge13_rema1000_babyogboern_1.jpg", customLabelRes, coco_format)

# fig = plt.figure(figsize=(1207/100,1489/100))
width, height = cocoImage.size

fig = plt.figure(figsize=(width/100,height/100))
fig.figimage(cocoImage)
plt.show()