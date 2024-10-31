import boto3
import boto3.session


class RekognitionDatasetCreator:
    def __init__(self, project_name: str, manifest_str: str, profile_name: str, region: str = 'eu-west-1'):
        """
        Initialize the Rekognition dataset creator

        Args:
            project_name: Name of the Custom Labels project
            manifest_str: manifest data
            region: AWS region to use
        """
        self.project_name = project_name
        self.manifest_str = manifest_str
        session = boto3.Session(profile_name=profile_name, region_name=region)
        self.rekognition = session.client('rekognition', region_name=region)

    def create_dataset(self, project_arn: str, dataset_type: str = 'TRAIN') -> str:
        """
        Create a new empty dataset in Amazon Rekognition Custom Labels

        Args:
            project_arn: Project the new dataset is added to
            dataset_type: Type of the dataset to create (TRAIN or TEST)

        Returns:
            dataset_arn: ARN of the created dataset
        """
        response = self.rekognition.create_dataset(
            DatasetType=dataset_type,
            ProjectArn=project_arn
        )
        return response['DatasetArn']

    def upload_annotations(self, dataset_arn: str, annotations: str) -> None:
        """
        Upload image annotations to the dataset

        Args:
            dataset_arn: ARN of the dataset to update
            annotations: s string with manifest entries
        """

        # Update dataset with manifest data
        self.rekognition.update_dataset_entries(
            DatasetArn=dataset_arn,
            Changes={
                'GroundTruth': annotations
            }
        )