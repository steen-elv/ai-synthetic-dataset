
import os
import cv2
import json
import numpy as np
from PIL import Image
from datetime import datetime
from pathlib import Path
import random


class OfferLayoutGenerator:
    def __init__(self, output_dir='ad_dataset'):
        self.output_dir = output_dir
        self.setup_directories()

        # COCO dataset structure
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

        # Layout settings
        self.canvas_size = (1200, 1600)  # A4-like proportion
        self.annotation_id = 1

        # Background settings
        self.bg_colors = [(255, 255, 255), (245, 245, 245), (240, 240, 240)]
        self.grid_patterns = ['none', 'dots', 'lines']

    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'images'), exist_ok=True)

    def create_background(self, size):
        """Create advertisement background with subtle patterns"""
        bg_color = random.choice(self.bg_colors)
        pattern = random.choice(self.grid_patterns)

        # Create base background
        bg = np.full((size[1], size[0], 3), bg_color, dtype=np.uint8)

        if pattern == 'dots':
            # Add subtle dot pattern
            spacing = random.randint(30, 50)
            for y in range(0, size[1], spacing):
                for x in range(0, size[0], spacing):
                    cv2.circle(bg, (x, y), 1, (220, 220, 220), -1)

        elif pattern == 'lines':
            # Add subtle line pattern
            spacing = random.randint(40, 60)
            color = (230, 230, 230)
            for y in range(0, size[1], spacing):
                cv2.line(bg, (0, y), (size[0], y), color, 1)
            for x in range(0, size[0], spacing):
                cv2.line(bg, (x, 0), (x, size[1]), color, 1)

        return bg

    def place_offers(self, offers, min_offers=2, max_offers=6):
        """Create advertisement layout with multiple offers"""
        # Create background
        canvas = self.create_background(self.canvas_size)

        # Randomly select number of offers to include
        num_offers = random.randint(min_offers, min(max_offers, len(offers)))
        selected_offers = random.sample(offers, num_offers)

        # Calculate grid-like layout
        rows = int(np.ceil(np.sqrt(num_offers)))
        cols = int(np.ceil(num_offers / rows))
        cell_width = self.canvas_size[0] // cols
        cell_height = self.canvas_size[1] // rows

        annotations = []
        used_positions = []

        for i, offer_path in enumerate(selected_offers):
            try:
                # Load and resize offer image
                offer_img = cv2.imread(str(offer_path))
                if offer_img is None:
                    continue

                # Calculate base position in grid
                row = i // cols
                col = i % cols

                # Add random offset
                offset_x = random.randint(-20, 20)
                offset_y = random.randint(-20, 20)

                # Calculate position and size
                max_width = int(cell_width * 0.9)
                max_height = int(cell_height * 0.9)

                # Resize offer while maintaining aspect ratio
                h, w = offer_img.shape[:2]
                scale = min(max_width / w, max_height / h)
                new_width = int(w * scale)
                new_height = int(h * scale)
                offer_img = cv2.resize(offer_img, (new_width, new_height))

                # Calculate position
                x = col * cell_width + (cell_width - new_width) // 2 + offset_x
                y = row * cell_height + (cell_height - new_height) // 2 + offset_y

                # Ensure offer stays within canvas bounds
                x = max(0, min(x, self.canvas_size[0] - new_width))
                y = max(0, min(y, self.canvas_size[1] - new_height))

                # Check for overlap with existing offers
                bbox = [x, y, x + new_width, y + new_height]
                overlap = False
                for used_bbox in used_positions:
                    if self.check_overlap(bbox, used_bbox):
                        overlap = True
                        break

                if not overlap:
                    # Place offer on canvas
                    canvas[y:y + new_height, x:x + new_width] = offer_img

                    # Create COCO annotation
                    annotation = {
                        "id": self.annotation_id,
                        "image_id": len(self.coco_dataset["images"]) + 1,
                        "category_id": 1,
                        "bbox": [x, y, new_width, new_height],
                        "area": new_width * new_height,
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
        """Generate dataset of advertisement layouts"""
        # Get all offer images
        offer_paths = []
        for ext in ['.jpg', '.jpeg', '.png']:
            offer_paths.extend(Path(input_folder).rglob(f'*{ext}'))

        print(f"Found {len(offer_paths)} offer images")

        for i in range(num_layouts):
            # Generate layout
            canvas, annotations = self.place_offers(offer_paths)

            # Save image
            image_filename = f'ad_{i + 1:06d}.jpg'
            image_path = os.path.join(self.output_dir, 'images', image_filename)
            cv2.imwrite(image_path, canvas)

            # Add image info to COCO dataset
            image_info = {
                "id": i + 1,
                "file_name": image_filename,
                "width": self.canvas_size[0],
                "height": self.canvas_size[1],
                "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.coco_dataset["images"].append(image_info)

            # Add annotations
            self.coco_dataset["annotations"].extend(annotations)

            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1} layouts")

        # Save COCO annotations
        annotation_path = os.path.join(self.output_dir, 'annotations.json')
        with open(annotation_path, 'w') as f:
            json.dump(self.coco_dataset, f, indent=2)

        print("\nDataset generation complete!")
        print(f"Generated {num_layouts} layouts")
        print(f"Total annotations: {len(self.coco_dataset['annotations'])}")


def generate_offer_dataset(input_folder, output_dir='ad_dataset', num_layouts=100):
    """Convenience function to generate offer detection dataset"""
    generator = OfferLayoutGenerator(output_dir)
    generator.generate_dataset(input_folder, num_layouts)


# Generate dataset from folder of offer images
input_folder = "/Users/steene/PycharmProjects/RekognitionExperiment/mt-input3"
generate_offer_dataset(input_folder, output_dir="ad_dataset", num_layouts=100)