import json
from datetime import datetime, UTC
from typing import Dict, List


def convert_coco_to_rekognition(coco_data: Dict,
                                bucket: str,
                                prefix: str,
                                class_name: str = "offer") -> str:
    """
    Convert COCO format annotations to AWS Rekognition Custom Labels manifest format

    Args:
        coco_data: Dictionary containing COCO format annotations
        bucket: S3 bucket name where images are stored
        prefix: S3 prefix (folder path) where images are stored
        class_name: Name of the class/category for the annotations

    Returns:
        str: Manifest data as newline-separated JSON strings
    """
    # Create image lookup dictionary
    images_lookup = {img['id']: img for img in coco_data['images']}

    # Store manifest entries
    manifest_entries = []

    # Group annotations by image_id
    image_annotations: Dict[int, List] = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)

    # Create manifest entry for each image
    for image_id, annotations in image_annotations.items():
        if image_id not in images_lookup:
            continue

        image_info = images_lookup[image_id]

        # Create annotations list
        label_annotations = []
        objects_metadata = []

        for idx, ann in enumerate(annotations):
            # COCO bbox format is [x, y, width, height]
            bbox = ann['bbox']

            label_annotations.append({
                "class_id": ann['category_id'],
                "top": int(bbox[1]),  # y
                "left": int(bbox[0]),  # x
                "width": int(bbox[2]),  # width
                "height": int(bbox[3])  # height
            })

            objects_metadata.append({
                "confidence": 1
            })

        # Create class map with all category IDs used in this image
        class_map = {
            str(ann['category_id']): class_name
            for ann in annotations
        }

        # Create manifest entry
        entry = {
            "source-ref": f"s3://{bucket}/{prefix}/{image_info['file_name']}",
            "bounding-box": {
                "image_size": [{
                    "width": image_info['width'],
                    "height": image_info['height'],
                    "depth": 3  # Assuming RGB images
                }],
                "annotations": label_annotations
            },
            "bounding-box-metadata": {
                "objects": objects_metadata,
                "class-map": class_map,
                "type": "groundtruth/object-detection",
                "human-annotated": "yes",
                "creation-date": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
                "job-name": "converted-from-coco"
            }
        }

        manifest_entries.append(json.dumps(entry))

    return '\n'.join(manifest_entries)


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
    with open('annotation_test.json', 'r') as fr:
     coco_data = json.load(fr)


    manifest_data = convert_coco_to_rekognition(
        coco_data=coco_data,
        bucket="your-bucket",
        prefix="your-prefix",
        class_name="offer"  # or whatever class name you want to use
    )

    # Print first entry as example
    print("First manifest entry:")
    print(manifest_data.split('\n')[0])

    # Optional: Save to file
    with open('rekognition_manifest.json', 'w') as f:
     f.write(manifest_data)