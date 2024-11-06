
import os
import cv2
import json
import numpy as np
from PIL import Image, ImageDraw
from datetime import datetime
from pathlib import Path
import random
import math


class EnhancedOfferLayoutGenerator:
    def __init__(self, output_dir='ad_dataset'):
        self.output_dir = output_dir
        # Store dimensions separately for clarity
        self.width = 1200
        self.height = 1600
        self.setup_directories()

        self.coco_dataset = {
            "info": {
                "year": datetime.now().year,
                "version": "1.0",
                "description": "Retail Offer Advertisement Dataset",
                "contributor": "Generated Dataset",
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "offer", "supercategory": "none"}]
        }
        self.annotation_id = 1

    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'images'), exist_ok=True)

    def create_complex_background(self):
        """Create complex background with correct dimensions"""
        # Create background with height, width order for numpy
        bg = np.full((self.height, self.width, 3), 240, dtype=np.uint8)

        # Add subtle noise
        noise = np.random.normal(0, 5, (self.height, self.width, 3))
        bg = np.clip(bg + noise, 0, 255).astype(np.uint8)

        # Add grid pattern
        grid_spacing = random.randint(30, 50)
        color = (220, 220, 220)

        # Vertical lines
        for x in range(0, self.width, grid_spacing):
            bg[:, x:x + 1] = color

        # Horizontal lines
        for y in range(0, self.height, grid_spacing):
            bg[y:y + 1, :] = color

        return bg

    def place_offers(self, offers, min_offers=2, max_offers=8):
        """Place offers on the canvas with corrected dimensions"""
        # Create background
        canvas = self.create_complex_background()

        # Select offers
        num_offers = random.randint(min_offers, min(max_offers, len(offers)))
        selected_offers = random.sample(offers, num_offers)

        annotations = []
        used_positions = []

        # Calculate grid layout
        rows = int(np.ceil(np.sqrt(num_offers)))
        cols = int(np.ceil(num_offers / rows))
        cell_width = self.width // cols
        cell_height = self.height // rows

        for i, offer_path in enumerate(selected_offers):
            try:
                # Load offer image
                offer_img = cv2.imread(str(offer_path))
                if offer_img is None:
                    continue

                # Calculate grid position
                row = i // cols
                col = i % cols

                # Scale offer to fit cell
                orig_height, orig_width = offer_img.shape[:2]
                scale = min(
                    (cell_width * 0.8) / orig_width,
                    (cell_height * 0.8) / orig_height
                )

                new_width = int(orig_width * scale)
                new_height = int(orig_height * scale)

                if new_width <= 0 or new_height <= 0:
                    continue

                # Resize offer
                offer_img = cv2.resize(offer_img, (new_width, new_height))

                # Calculate position with centering
                x = col * cell_width + (cell_width - new_width) // 2
                y = row * cell_height + (cell_height - new_height) // 2

                # Add random offset
                x += random.randint(-20, 20)
                y += random.randint(-20, 20)

                # Ensure within bounds
                x = max(0, min(x, self.width - new_width))
                y = max(0, min(y, self.height - new_height))

                # Double-check dimensions
                if (y + new_height <= self.height and
                        x + new_width <= self.width):

                    # Check overlap
                    bbox = [x, y, x + new_width, y + new_height]
                    overlap = any(self.check_overlap(bbox, used_bbox)
                                  for used_bbox in used_positions)

                    if not overlap:
                        # Get region of interest dimensions
                        roi_height = min(new_height, self.height - y)
                        roi_width = min(new_width, self.width - x)

                        # Ensure offer image is cropped to match ROI if needed
                        offer_roi = offer_img[:roi_height, :roi_width]

                        # Place offer
                        canvas[y:y + roi_height, x:x + roi_width] = offer_roi

                        # Create annotation
                        annotation = {
                            "id": self.annotation_id,
                            "image_id": len(self.coco_dataset["images"]) + 1,
                            "category_id": 1,
                            "bbox": [x, y, roi_width, roi_height],
                            "area": roi_width * roi_height,
                            "segmentation": [],
                            "iscrowd": 0
                        }
                        annotations.append(annotation)
                        used_positions.append(bbox)
                        self.annotation_id += 1

            except Exception as e:
                print(f"Error processing offer {offer_path}: {str(e)}")
                continue

        return canvas, annotations

    @staticmethod
    def check_overlap(bbox1, bbox2, threshold=0):
        """Check if two bounding boxes overlap"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        return not (x1_max < x2_min - threshold or
                    x1_min > x2_max + threshold or
                    y1_max < y2_min - threshold or
                    y1_min > y2_max + threshold)

    def generate_dataset(self, input_folder, num_layouts=100):
        """Generate dataset with dimension-safe image handling"""
        # Get all offer images
        offer_paths = []
        for ext in ['.jpg', '.jpeg', '.png']:
            offer_paths.extend(Path(input_folder).rglob(f'*{ext}'))

        if not offer_paths:
            raise ValueError(f"No images found in {input_folder}")

        print(f"Found {len(offer_paths)} offer images")

        for i in range(num_layouts):
            try:
                # Generate layout
                canvas, annotations = self.place_offers(offer_paths)

                if len(annotations) > 0:
                    # Save image
                    image_filename = f'ad_{i + 1:06d}.jpg'
                    image_path = os.path.join(self.output_dir, 'images', image_filename)
                    cv2.imwrite(image_path, canvas)

                    # Add image info
                    image_info = {
                        "id": i + 1,
                        "file_name": image_filename,
                        "width": self.width,
                        "height": self.height,
                        "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    self.coco_dataset["images"].append(image_info)
                    self.coco_dataset["annotations"].extend(annotations)

                    if (i + 1) % 10 == 0:
                        print(f"Generated {i + 1} layouts")

            except Exception as e:
                print(f"Error generating layout {i + 1}: {str(e)}")
                continue

        # Save annotations
        annotation_path = os.path.join(self.output_dir, 'annotations.json')
        with open(annotation_path, 'w') as f:
            json.dump(self.coco_dataset, f, indent=2)

        print(f"\nDataset generation complete!")
        print(f"Generated {len(self.coco_dataset['images'])} layouts")
        print(f"Total annotations: {len(self.coco_dataset['annotations'])}")


def generate_offer_dataset(input_folder, output_dir='ad_dataset', num_layouts=100):
    """Convenience function to generate offer detection dataset"""
    generator = EnhancedOfferLayoutGenerator(output_dir)

    print("Starting dataset generation...")
    print(f"Input folder: {input_folder}")
    print(f"Output directory: {output_dir}")
    print(f"Number of layouts to generate: {num_layouts}")

    try:
        generator.generate_dataset(input_folder, num_layouts)
        return True
    except Exception as e:
        print(f"Error during dataset generation: {str(e)}")
        return False




if __name__ == "__main__":
    # Example usage
    input_folder = "/Users/steene/PycharmProjects/RekognitionExperiment/mt-input3"
    generate_offer_dataset(input_folder, output_dir="ad_dataset", num_layouts=5000)
