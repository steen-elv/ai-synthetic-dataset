import json
from datetime import datetime, UTC
from typing import Dict, List


def convert_coco_to_rekognition(coco_data: Dict,
                                available_images: List[str],
                                bucket: str,
                                prefix: str,
                                class_name: str = "offer") -> str:
    """
    Convert COCO format annotations to AWS Rekognition Custom Labels manifest format

    Args:
        coco_data: Dictionary containing COCO format annotations
        available_images: List containing the available images in the bucket folder
        bucket: S3 bucket name where images are stored
        prefix: S3 prefix (folder path) where images are stored
        class_name: Name of the class/category for the annotations

    Returns:
        str: Manifest data as newline-separated JSON strings
    """
    # Create image lookup dictionary
    images_lookup = {img['id']: img for img in coco_data['images']}

    available_images_lookup = {fn: fn for fn in available_images}


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

        if image_info['file_name'] not in available_images_lookup:
            print(f'Skipping {image_info["file_name"]}')
            continue

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
