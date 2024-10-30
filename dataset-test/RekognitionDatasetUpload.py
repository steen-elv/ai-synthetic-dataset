import boto3
import json
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
                    "labels": [
                        {
                            "name": "label_name",
                            "bbox": {
                                "left": float,
                                "top": float,
                                "width": float,
                                "height": float
                            }
                        }
                    ]
                }
            bucket: S3 bucket containing the images
            prefix: Prefix path in the S3 bucket
        """
        manifest = self._create_manifest(annotations, bucket, prefix)
        manifest_key = f'{prefix}/manifest.json'

        # Upload manifest to S3
        self.s3.put_object(
            Bucket=bucket,
            Key=manifest_key,
            Body=json.dumps(manifest)
        )

        # Update dataset with manifest
        self.rekognition.update_dataset_entries(
            DatasetArn=dataset_arn,
            Changes=[{
                'GroundTruth': {
                    'S3Object': {
                        'Bucket': bucket,
                        'Name': manifest_key
                    }
                }
            }]
        )

    def _get_project_arn(self) -> str:
        """Get the project ARN from the project name"""
        response = self.rekognition.describe_project(
            ProjectArn=f'arn:aws:rekognition:{self.rekognition.meta.region_name}:{self.rekognition.meta.account_id}:project/{self.project_name}/1.0'
        )
        return response['ProjectArn']

    def _create_manifest(self, annotations: List[Dict], bucket: str, prefix: str) -> List[Dict]:
        """Convert annotations to Rekognition manifest format"""
        manifest = []
        for ann in annotations:
            entry = {
                'source-ref': f's3://{bucket}/{prefix}/{ann["image_key"]}',
                'custom-labels': {
                    'labels': []
                }
            }

            for label in ann['labels']:
                label_entry = {
                    'name': label['name'],
                    'bbox': {
                        'left': label['bbox']['left'],
                        'top': label['bbox']['top'],
                        'width': label['bbox']['width'],
                        'height': label['bbox']['height']
                    }
                }
                entry['custom-labels']['labels'].append(label_entry)

            manifest.append(entry)

        return manifest



# Example usage
creator = RekognitionDatasetCreator('my-project')

# Create a new dataset
dataset_arn = creator.create_dataset('my-dataset')

# Prepare your annotations
custom_annotations = [
    {
        "image_key": "image1.jpg",
        "labels": [
            {
                "name": "car",
                "bbox": {
                    "left": 0.1,
                    "top": 0.2,
                    "width": 0.3,
                    "height": 0.4
                }
            }
        ]
    }
]

# Upload annotations
creator.upload_annotations(
    dataset_arn=dataset_arn,
    annotations=custom_annotations,
    bucket='my-bucket',
    prefix='my-images'
)