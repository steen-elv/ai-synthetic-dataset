
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

        self.canvas_size = (1200, 1600)
        self.annotation_id = 1

        # Enhanced background patterns
        self.background_types = [
            'noise', 'gradient', 'pattern', 'texture',
            'scattered', 'geometric', 'grid', 'waves'
        ]

        # Layout strategies
        self.layout_strategies = [
            'grid', 'diagonal', 'circular', 'random',
            'clustered', 'spiral', 'columns', 'mosaic'
        ]

    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'images'), exist_ok=True)

    def create_complex_background(self, size):
        """Create complex background with various patterns and noise"""
        bg_type = random.choice(self.background_types)
        bg = np.zeros((size[1], size[0], 3), dtype=np.uint8)

        if bg_type == 'noise':
            # Complex noise pattern
            noise_types = ['gaussian', 'speckle', 'salt_pepper']
            noise_type = random.choice(noise_types)

            if noise_type == 'gaussian':
                # Generate noise and clip to valid range
                noise = np.random.normal(220, 20, (size[1], size[0], 3))
                bg = np.clip(noise, 0, 255).astype(np.uint8)
            elif noise_type == 'speckle':
                # Use uniform noise instead of rayleigh for better control
                noise = np.random.uniform(200, 240, (size[1], size[0], 3))
                bg = noise.astype(np.uint8)
            else:  # salt_pepper
                bg.fill(220)
                mask = np.random.random(size[:-1]) < 0.02
                bg[mask] = random.choice([200, 240])

        elif bg_type == 'gradient':
            # Simplified gradient calculation
            angle = random.uniform(0, 2 * np.pi)
            for y in range(size[1]):
                for x in range(size[0]):
                    value = (math.cos(angle) * x + math.sin(angle) * y) / (size[0] + size[1])
                    # Ensure value stays within 0-255 range
                    color = int(np.clip(220 + 35 * value, 0, 255))
                    bg[y, x] = [color] * 3

        elif bg_type == 'pattern':
            # Geometric patterns
            bg.fill(240)
            pattern_size = random.randint(20, 50)
            for y in range(0, size[1], pattern_size):
                for x in range(0, size[0], pattern_size):
                    if (x + y) % (pattern_size * 2) == 0:
                        cv2.rectangle(bg, (x, y), (x + pattern_size, y + pattern_size),
                                      (220, 220, 220), -1)

        elif bg_type == 'texture':
            # Random texture with controlled range
            texture_scale = random.randint(2, 5)
            small_noise = np.random.uniform(
                200, 240,
                (size[1] // texture_scale, size[0] // texture_scale, 3)
            ).astype(np.uint8)
            bg = cv2.resize(small_noise, (size[0], size[1]))

        elif bg_type == 'scattered':
            # Scattered shapes
            bg.fill(240)
            for _ in range(300):
                shape_type = random.choice(['circle', 'rectangle', 'line'])
                x = random.randint(0, size[0] - 1)  # Ensure within bounds
                y = random.randint(0, size[1] - 1)  # Ensure within bounds
                color = random.randint(200, 240)

                if shape_type == 'circle':
                    radius = random.randint(2, 8)
                    cv2.circle(bg, (x, y), radius, (color, color, color), -1)
                elif shape_type == 'rectangle':
                    w = random.randint(4, 12)
                    h = random.randint(4, 12)
                    # Ensure rectangle stays within image bounds
                    x2 = min(x + w, size[0] - 1)
                    y2 = min(y + h, size[1] - 1)
                    cv2.rectangle(bg, (x, y), (x2, y2), (color, color, color), -1)
                else:  # line
                    length = random.randint(10, 30)
                    angle = random.uniform(0, 2 * np.pi)
                    end_x = int(np.clip(x + length * math.cos(angle), 0, size[0] - 1))
                    end_y = int(np.clip(y + length * math.sin(angle), 0, size[1] - 1))
                    cv2.line(bg, (x, y), (end_x, end_y), (color, color, color), 1)

        elif bg_type == 'geometric':
            # Complex geometric background
            bg.fill(240)
            for _ in range(50):
                num_points = random.randint(3, 6)
                points = np.random.rand(num_points, 2)
                # Ensure points are within image bounds
                points = (points * np.array([size[0] - 1, size[1] - 1])).astype(np.int32)
                color = random.randint(220, 240)
                cv2.fillPoly(bg, [points], (color, color, color))

        elif bg_type == 'waves':
            # Wave pattern with controlled range
            for y in range(size[1]):
                for x in range(size[0]):
                    wave = math.sin(x / 30) + math.cos(y / 20)
                    # Ensure color stays within valid range
                    color = int(np.clip(220 + 20 * wave, 0, 255))
                    bg[y, x] = [color] * 3

        # Add subtle noise overlay with controlled range
        noise_overlay = np.random.normal(0, 2, bg.shape)
        bg = np.clip(bg.astype(np.float32) + noise_overlay, 0, 255).astype(np.uint8)

        return bg

    def get_offer_placement_strategy(self, num_offers, strategy):
        """Get offer positions based on different layout strategies"""
        positions = []
        margin = 50
        usable_width = self.canvas_size[0] - 2 * margin
        usable_height = self.canvas_size[1] - 2 * margin

        if strategy == 'grid':
            rows = int(np.ceil(np.sqrt(num_offers)))
            cols = int(np.ceil(num_offers / rows))
            cell_width = usable_width / cols
            cell_height = usable_height / rows

            for i in range(num_offers):
                row = i // cols
                col = i % cols
                x = margin + col * cell_width + random.randint(-20, 20)
                y = margin + row * cell_height + random.randint(-20, 20)
                positions.append((x, y))

        elif strategy == 'diagonal':
            step = 1.0 / (num_offers - 1) if num_offers > 1 else 0.5
            for i in range(num_offers):
                t = i * step
                x = margin + t * usable_width + random.randint(-30, 30)
                y = margin + t * usable_height + random.randint(-30, 30)
                positions.append((x, y))

        elif strategy == 'circular':
            center_x = self.canvas_size[0] / 2
            center_y = self.canvas_size[1] / 2
            radius = min(usable_width, usable_height) / 3

            for i in range(num_offers):
                angle = (i * 2 * np.pi / num_offers) + random.uniform(-0.2, 0.2)
                x = center_x + radius * math.cos(angle) + random.randint(-30, 30)
                y = center_y + radius * math.sin(angle) + random.randint(-30, 30)
                positions.append((x, y))

        elif strategy == 'clustered':
            # Create 2-3 cluster centers
            num_clusters = random.randint(2, 3)
            cluster_centers = []
            for _ in range(num_clusters):
                x = random.randint(margin, self.canvas_size[0] - margin)
                y = random.randint(margin, self.canvas_size[1] - margin)
                cluster_centers.append((x, y))

            for _ in range(num_offers):
                center = random.choice(cluster_centers)
                x = center[0] + random.gauss(0, 100)
                y = center[1] + random.gauss(0, 100)
                x = max(margin, min(x, self.canvas_size[0] - margin))
                y = max(margin, min(y, self.canvas_size[1] - margin))
                positions.append((x, y))

        elif strategy == 'spiral':
            center_x = self.canvas_size[0] / 2
            center_y = self.canvas_size[1] / 2

            for i in range(num_offers):
                t = i / num_offers * 4 * np.pi
                r = t * min(usable_width, usable_height) / (8 * np.pi)
                x = center_x + r * math.cos(t) + random.randint(-20, 20)
                y = center_y + r * math.sin(t) + random.randint(-20, 20)
                positions.append((x, y))

        else:  # random
            for _ in range(num_offers):
                x = random.randint(margin, self.canvas_size[0] - margin)
                y = random.randint(margin, self.canvas_size[1] - margin)
                positions.append((x, y))

        return positions

    def place_offers(self, offers, min_offers=2, max_offers=8):
        """Create advertisement layout with multiple offers"""
        # Create complex background
        canvas = self.create_complex_background(self.canvas_size)

        # Select number of offers and layout strategy
        num_offers = random.randint(min_offers, min(max_offers, len(offers)))
        selected_offers = random.sample(offers, num_offers)
        strategy = random.choice(self.layout_strategies)

        # Get initial positions
        positions = self.get_offer_placement_strategy(num_offers, strategy)

        annotations = []
        used_positions = []

        for offer_path, initial_pos in zip(selected_offers, positions):
            try:
                # Load offer image
                offer_img = cv2.imread(str(offer_path))
                if offer_img is None:
                    continue

                # Random scaling
                scale = random.uniform(0.5, 1.0)
                h, w = offer_img.shape[:2]
                new_width = int(w * scale)
                new_height = int(h * scale)
                offer_img = cv2.resize(offer_img, (new_width, new_height))

                # Random rotation (slight)
                if random.random() < 0.3:
                    angle = random.uniform(-5, 5)
                    matrix = cv2.getRotationMatrix2D((new_width // 2, new_height // 2), angle, 1.0)
                    offer_img = cv2.warpAffine(offer_img, matrix, (new_width, new_height))

                # Position with initial placement strategy
                x, y = initial_pos
                x = int(x - new_width // 2)
                y = int(y - new_height // 2)

                # Ensure within bounds
                x = max(0, min(x, self.canvas_size[0] - new_width))
                y = max(0, min(y, self.canvas_size[1] - new_height))

                # Check overlap
                bbox = [x, y, x + new_width, y + new_height]
                overlap = False
                for used_bbox in used_positions:
                    if self.check_overlap(bbox, used_bbox, threshold=20):
                        overlap = True
                        break

                if not overlap:
                    # Place offer on canvas
                    canvas[y:y + new_height, x:x + new_width] = offer_img

                    # Create annotation
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

            # Add image info
            image_info = {
                "id": i + 1,
                "file_name": image_filename,
                "width": self.canvas_size[0],
                "height": self.canvas_size[1],
                "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.coco_dataset["images"].append(image_info)
            self.coco_dataset["annotations"].extend(annotations)


            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1} layouts")

                # Periodic save of annotations (backup)
                temp_annotation_path = os.path.join(self.output_dir, 'annotations_backup.json')
                with open(temp_annotation_path, 'w') as f:
                    json.dump(self.coco_dataset, f, indent=2)

        # Save final COCO annotations
        annotation_path = os.path.join(self.output_dir, 'annotations.json')
        with open(annotation_path, 'w') as f:
            json.dump(self.coco_dataset, f, indent=2)

        # Generate and save dataset statistics
        stats = {
            "total_layouts": num_layouts,
            "total_annotations": len(self.coco_dataset["annotations"]),
            "average_offers_per_layout": len(self.coco_dataset["annotations"]) / num_layouts,
            "completion_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        stats_path = os.path.join(self.output_dir, 'dataset_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print("\nDataset generation complete!")
        print(f"Generated {num_layouts} layouts")
        print(f"Total annotations: {len(self.coco_dataset['annotations'])}")
        print(f"Average offers per layout: {stats['average_offers_per_layout']:.2f}")


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
    generate_offer_dataset(input_folder, output_dir="ad_dataset", num_layouts=100)
