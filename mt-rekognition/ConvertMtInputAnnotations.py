import sys
sys.path.insert(0,"..")
from ConvertAnnotations import *

# Example usage
if __name__ == "__main__":
    # Your COCO data
    # coco_data = {
    #     "images": [
    #         {
    #             "file_name": "2024_uge14_riisrejser_hovedkatalogdec_apr_2.jpg",
    #             "width": 1143,
    #             "height": 1617,
    #             "id": 783
    #         }
    #         # ... other images ...
    #     ],
    #     "annotations": [
    #         {
    #             "image_id": 783,
    #             "bbox": [8, 11, 344, 502],
    #             "category_id": 1
    #         }
    #         # ... other annotations ...
    #     ]
    # }
    with open('annotations.json', 'r') as fr:
     coco_data = json.load(fr)


    manifest_data = convert_coco_to_rekognition(
        coco_data=coco_data,
        bucket="minetilbud-example-1",
        prefix="mixed",
        class_name="offer"  # or whatever class name you want to use
    )

    # Print first entry as example
    print("First manifest entry:")
    print(manifest_data.split('\n')[0])

    # Optional: Save to file
    with open('rekognition_manifest.json', 'w') as f:
     f.write(manifest_data)