import boto3
import json
from datetime import datetime
from typing import List, Dict


class RekognitionDatasetCreator:
    def __init__(self, project_name: str, region: str = 'eu-west-1'):
        """
        Initialize the Rekognition dataset creator

        Args:
            project_name: Name of the Custom Labels project
            region: AWS region to use
        """
        self.project_name = project_name
        self.rekognition = boto3.client('rekognition', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)

    def create_dataset(self, dataset_name: str) -> str:
        """
        Create a new dataset in Amazon Rekognition Custom Labels

        Args:
            dataset_name: Name of the dataset to create

        Returns:
            dataset_arn: ARN of the created dataset
        """
        response = self.rekognition.create_dataset(
            DatasetType='TRAIN',
            DatasetSource={
                'GroundTruth': {
                    'Format': 'JSON'
                }
            },
            ProjectArn=self._get_project_arn(),
            DatasetDescription=f'Training dataset for {self.project_name}'
        )
        return response['DatasetArn']

    def upload_annotations(self, dataset_arn: str, annotations: List[Dict], bucket: str, prefix: str) -> None:
        """
        Upload image annotations to the dataset

        Args:
            dataset_arn: ARN of the dataset to update
            annotations: List of annotation dictionaries in the format:
                {
                    "image_key": "path/to/image.jpg",
                    "image_size": {
                        "width": int,
                        "height": int,
                        "depth": int
                    },
                    "labels": [
                        {
                            "name": "label_name",
                            "class_id": int,
                            "bbox": {
                                "top": int,
                                "left": int,
                                "width": int,
                                "height": int
                            }
                        }
                    ]
                }
            bucket: S3 bucket containing the images
            prefix: Prefix path in the S3 bucket
        """
        manifest_entries = []

        for ann in annotations:
            # Create class map from labels
            class_map = {}
            label_annotations = []
            objects_metadata = []

            for label in ann['labels']:
                class_map[str(label['class_id'])] = label['name']

                # Add bounding box annotation
                label_annotations.append({
                    "class_id": label['class_id'],
                    "top": label['bbox']['top'],
                    "left": label['bbox']['left'],
                    "width": label['bbox']['width'],
                    "height": label['bbox']['height']
                })

                # Add corresponding metadata
                objects_metadata.append({
                    "confidence": 1
                })

            # Create manifest entry
            entry = {
                "source-ref": f"s3://{bucket}/{prefix}/{ann['image_key']}",
                "bounding-box": {
                    "image_size": [{
                        "width": ann['image_size']['width'],
                        "height": ann['image_size']['height'],
                        "depth": ann['image_size']['depth']
                    }],
                    "annotations": label_annotations
                },
                "bounding-box-metadata": {
                    "objects": objects_metadata,
                    "class-map": class_map,
                    "type": "groundtruth/object-detection",
                    "human-annotated": "yes",
                    "creation-date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
                    "job-name": self.project_name
                }
            }

            manifest_entries.append(json.dumps(entry))

        # Join entries with newlines
        manifest_data = '\n'.join(manifest_entries)

        # Update dataset with manifest data
        self.rekognition.update_dataset_entries(
            DatasetArn=dataset_arn,
            Changes={
                'GroundTruth': manifest_data
            }
        )

    def _get_project_arn(self) -> str:
        """Get the project ARN from the project name"""
        response = self.rekognition.describe_project(
            ProjectArn=f'arn:aws:rekognition:{self.rekognition.meta.region_name}:{self.rekognition.meta.account_id}:project/{self.project_name}/1.0'
        )
        return response['ProjectArn']