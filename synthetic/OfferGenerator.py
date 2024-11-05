import os
import json
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import shutil
from datetime import datetime


class OfferDatasetProcessor:
    def __init__(self, output_dir='processed_dataset'):
        self.output_dir = output_dir
        self.setup_directories()

        # Dataset statistics
        self.stats = {
            'total_images': 0,
            'processed_images': 0,
            'skipped_images': 0,
            'image_sizes': {},
            'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def setup_directories(self):
        """Create necessary directory structure"""
        dirs = ['images', 'annotations', 'metadata', 'splits']
        for dir_name in dirs:
            os.makedirs(os.path.join(self.output_dir, dir_name), exist_ok=True)

    def process_image_folder(self, input_folder, copy_original=True):
        """Process all images in the input folder"""
        print(f"Starting to process images from: {input_folder}")

        input_path = Path(input_folder)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

        # Get all image files
        image_files = [
            f for f in input_path.rglob("*")
            if f.suffix.lower() in image_extensions
        ]

        self.stats['total_images'] = len(image_files)
        print(f"Found {len(image_files)} images to process")

        for idx, image_path in enumerate(image_files, 1):
            try:
                # Load image
                img = Image.open(image_path)

                # Generate unique identifier
                image_id = f"offer_{idx:06d}"

                # Process image and create annotation
                self._process_single_image(img, image_id, image_path, copy_original)

                self.stats['processed_images'] += 1

                # Record image size statistics
                size_key = f"{img.width}x{img.height}"
                self.stats['image_sizes'][size_key] = self.stats['image_sizes'].get(size_key, 0) + 1

                # Progress update
                if idx % 100 == 0:
                    print(f"Processed {idx} images...")
                    self._save_stats()  # Periodic stats update

            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                self.stats['skipped_images'] += 1
                continue

        # Save final statistics
        self._save_stats()
        self._create_dataset_splits()
        print("\nProcessing complete!")
        print(f"Successfully processed: {self.stats['processed_images']} images")
        print(f"Skipped: {self.stats['skipped_images']} images")

    def _process_single_image(self, img, image_id, source_path, copy_original=True):
        """Process a single image and create its annotation"""
        # Prepare paths
        if copy_original:
            # Copy original image
            dest_path = os.path.join(self.output_dir, 'images', f"{image_id}{source_path.suffix}")
            shutil.copy2(source_path, dest_path)
        else:
            # Save as PNG with potential preprocessing
            dest_path = os.path.join(self.output_dir, 'images', f"{image_id}.png")
            img.save(dest_path)

        # Create basic annotation
        annotation = {
            'image_id': image_id,
            'file_name': os.path.basename(dest_path),
            'width': img.width,
            'height': img.height,
            'source_path': str(source_path),
            'date_processed': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Save annotation
        annotation_path = os.path.join(
            self.output_dir, 'annotations', f"{image_id}.json")
        with open(annotation_path, 'w') as f:
            json.dump(annotation, f, indent=2)

        return annotation

    def _save_stats(self):
        """Save processing statistics"""
        stats_path = os.path.join(self.output_dir, 'metadata', 'dataset_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)

    def _create_dataset_splits(self, train_ratio=0.8, val_ratio=0.1):
        """Create train/validation/test splits"""
        # Get all processed image IDs
        annotation_files = os.listdir(os.path.join(self.output_dir, 'annotations'))
        image_ids = [f.split('.')[0] for f in annotation_files]

        # Shuffle image IDs
        np.random.shuffle(image_ids)

        # Calculate split sizes
        total = len(image_ids)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)

        # Create splits
        splits = {
            'train': image_ids[:train_size],
            'validation': image_ids[train_size:train_size + val_size],
            'test': image_ids[train_size + val_size:]
        }

        # Save splits
        splits_path = os.path.join(self.output_dir, 'splits', 'dataset_splits.json')
        with open(splits_path, 'w') as f:
            json.dump(splits, f, indent=2)

    def create_dataset_info(self):
        """Create comprehensive dataset information file"""
        info = {
            'name': 'Retail Offers Dataset',
            'description': 'Dataset of retail product offers and advertisements',
            'date_created': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'statistics': self.stats,
            'directory_structure': {
                'images': 'Original processed offer images',
                'annotations': 'JSON annotations for each image',
                'metadata': 'Dataset statistics and processing information',
                'splits': 'Train/validation/test split definitions'
            }
        }

        info_path = os.path.join(self.output_dir, 'metadata', 'dataset_info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)

    def verify_dataset(self):
        """Verify dataset integrity"""
        verification_results = {
            'missing_images': [],
            'missing_annotations': [],
            'mismatched_pairs': []
        }

        # Check all images have annotations and vice versa
        image_files = set(os.listdir(os.path.join(self.output_dir, 'images')))
        annotation_files = set(os.listdir(os.path.join(self.output_dir, 'annotations')))

        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0]
            if f"{base_name}.json" not in annotation_files:
                verification_results['missing_annotations'].append(img_file)

        for ann_file in annotation_files:
            base_name = os.path.splitext(ann_file)[0]
            if not any(f.startswith(base_name) for f in image_files):
                verification_results['missing_images'].append(ann_file)

        # Save verification results
        verify_path = os.path.join(self.output_dir, 'metadata', 'verification_results.json')
        with open(verify_path, 'w') as f:
            json.dump(verification_results, f, indent=2)

        return verification_results


def process_retail_dataset(input_folder, output_dir='processed_dataset'):
    """Convenience function to process an entire retail offer dataset"""
    processor = OfferDatasetProcessor(output_dir)

    print("Starting dataset processing...")
    processor.process_image_folder(input_folder)

    print("Creating dataset information...")
    processor.create_dataset_info()

    print("Verifying dataset integrity...")
    verification_results = processor.verify_dataset()

    print("\nDataset processing complete!")
    print(f"Output directory: {output_dir}")

    return verification_results


# Process entire folder of offer images
input_folder = "/Users/steene/PycharmProjects/RekognitionExperiment/mt-input3"
verification_results = process_retail_dataset(input_folder, output_dir="processed_dataset")