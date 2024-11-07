
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
    def __init__(self, input_folder, non_offer_folder, output_dir='ad_dataset', start_with=1):
        self.start_with = start_with
        self.non_offer_folder = non_offer_folder
        self.input_folder = input_folder
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
        """Create more realistic complex background"""
        bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Create multiple layers of visual elements

        # Base layer - gradient or multi-color background
        start_color = np.random.randint(180, 255, 3)
        end_color = np.random.randint(180, 255, 3)
        for y in range(self.height):
            ratio = y / self.height
            bg[y] = start_color * (1 - ratio) + end_color * ratio

        # Add organic patterns
        num_shapes = random.randint(3, 8)
        for _ in range(num_shapes):
            shape_color = np.random.randint(160, 255, 3)
            center_x = random.randint(0, self.width)
            center_y = random.randint(0, self.height)
            radius = random.randint(50, 200)

            y, x = np.ogrid[-center_y:self.height - center_y, -center_x:self.width - center_x]
            mask = x * x + y * y <= radius * radius

            # Apply with transparency
            alpha = random.uniform(0.1, 0.3)
            bg[mask] = bg[mask] * (1 - alpha) + shape_color * alpha

        # Add noise with varying intensity
        noise_intensity = random.uniform(2, 8)
        noise = np.random.normal(0, noise_intensity, (self.height, self.width, 3))
        bg = np.clip(bg + noise, 0, 255).astype(np.uint8)

        # Optional: Add texture patterns
        if random.random() > 0.5:
            texture_scale = random.randint(10, 30)
            texture = np.random.randint(0, 30, (self.height // texture_scale, self.width // texture_scale, 3))
            texture = cv2.resize(texture.astype(float), (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            bg = np.clip(bg + texture, 0, 255).astype(np.uint8)

        return self.add_non_offer_elements(bg)

    def add_non_offer_elements(self, background):
        """Add product images and design elements that aren't offers"""
        # Add decorative shapes
        num_elements = random.randint(2, 5)
        for _ in range(num_elements):
            # Random geometric shapes
            shape_type = random.choice(['rectangle', 'circle', 'line'])
            color = tuple(np.random.randint(0, 255, 3).tolist())

            if shape_type == 'rectangle':
                x1 = random.randint(0, self.width - 100)
                y1 = random.randint(0, self.height - 100)
                w = random.randint(50, 200)
                h = random.randint(50, 200)
                cv2.rectangle(background, (x1, y1), (x1 + w, y1 + h), color, -1)

        # Add non-offer product images if available
        # This assumes you have a collection of non-offer product images
        non_offer_paths = []
        for ext in ['.jpg', '.jpeg', '.png']:
            non_offer_paths.extend(Path(self.non_offer_folder).rglob(f'*{ext}'))

        background = self.add_non_offer_products(background, non_offer_paths, min_products=1, max_products=5)

        return background

    def add_non_offer_products(self, background, non_offer_images, min_products=2, max_products=4):
        """
        Add non-offer product images to the background

        Args:
            background (numpy.ndarray): The background image
            non_offer_images (list): List of non-offer product image paths or loaded images
            min_products (int): Minimum number of products to add
            max_products (int): Maximum number of products to add

        Returns:
            numpy.ndarray: Background with added product images
        """
        # Create a copy of the background to avoid modifying the original
        result = background.copy()

        # Randomly decide how many products to add
        num_products = random.randint(min_products, max_products)

        # Keep track of used positions to avoid overlap
        used_positions = []

        for _ in range(num_products):
            # Randomly select a product image
            product_path = random.choice(non_offer_images)

            # Load the image
            product = cv2.imread(product_path, cv2.IMREAD_UNCHANGED)

            # Random scaling factor
            scale_factor = random.uniform(0.1, 0.3)

            # Calculate new dimensions while maintaining aspect ratio
            new_width = int(result.shape[1] * scale_factor)
            aspect_ratio = product.shape[1] / product.shape[0]
            new_height = int(new_width / aspect_ratio)

            # Resize product
            product_resized = cv2.resize(product, (new_width, new_height))

            # Convert to RGBA if not already
            if product_resized.shape[2] == 3:
                product_resized = cv2.cvtColor(product_resized, cv2.COLOR_BGR2BGRA)

            # Try to find a suitable position (max 10 attempts)
            position_found = False
            for _ in range(10):
                # Random position
                x = random.randint(0, result.shape[1] - new_width)
                y = random.randint(0, result.shape[0] - new_height)

                # Check if position overlaps with used positions
                overlap = False
                for used_pos in used_positions:
                    used_x, used_y, used_w, used_h = used_pos
                    if (x < used_x + used_w and x + new_width > used_x and
                            y < used_y + used_h and y + new_height > used_y):
                        overlap = True
                        break

                if not overlap:
                    position_found = True
                    used_positions.append((x, y, new_width, new_height))
                    break

            if not position_found:
                continue

            # Create mask from alpha channel
            alpha = product_resized[:, :, 3] / 255.0

            # Add some random rotation
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                rotation_matrix = cv2.getRotationMatrix2D((new_width / 2, new_height / 2), angle, 1.0)
                product_resized = cv2.warpAffine(product_resized, rotation_matrix, (new_width, new_height))
                alpha = cv2.warpAffine(alpha, rotation_matrix, (new_width, new_height))

            # brightness/contrast adjustment
            if random.random() > 0.5:
                alpha = random.uniform(0.8, 1.2)  # Contrast
                beta = random.randint(-30, 30)  # Brightness
                product_resized = cv2.convertScaleAbs(product_resized, alpha=alpha, beta=beta)

            # Add perspective transformation
            if random.random() > 0.7:
                pts1 = np.float32([[0, 0], [new_width, 0], [0, new_height], [new_width, new_height]])
                pts2 = np.float32([
                    [random.randint(-10, 10), random.randint(-10, 10)],
                    [new_width - random.randint(-10, 10), random.randint(-10, 10)],
                    [random.randint(-10, 10), new_height - random.randint(-10, 10)],
                    [new_width - random.randint(-10, 10), new_height - random.randint(-10, 10)]
                ])
                perspective_transform = cv2.getPerspectiveTransform(pts1, pts2)
                product_resized = cv2.warpPerspective(product_resized, perspective_transform, (new_width, new_height))

            # Blend product with background
            for c in range(3):
                result[y:y + new_height, x:x + new_width, c] = (
                        (1 - alpha) * result[y:y + new_height, x:x + new_width, c] +
                        alpha * product_resized[:, :, c]
                )

        return result

    def apply_augmentation(self, image):
        """Apply realistic augmentation to make synthetic images more natural"""
        # Random brightness and contrast
        alpha = random.uniform(0.8, 1.2)  # Contrast
        beta = random.randint(-30, 30)  # Brightness
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        # Random blur
        if random.random() > 0.7:
            blur_size = random.choice([3, 5, 7])
            image = cv2.GaussianBlur(image, (blur_size, blur_size), 0)

        # Perspective transform
        if random.random() > 0.8:
            pts1 = np.float32([[0, 0], [image.shape[1], 0],
                               [0, image.shape[0]], [image.shape[1], image.shape[0]]])
            pts2 = np.float32([[random.randint(-20, 20), random.randint(-20, 20)],
                               [image.shape[1] - random.randint(-20, 20), random.randint(-20, 20)],
                               [random.randint(-20, 20), image.shape[0] - random.randint(-20, 20)],
                               [image.shape[1] - random.randint(-20, 20), image.shape[0] - random.randint(-20, 20)]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            image = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))

        # JPEG compression simulation
        if random.random() > 0.6:
            quality = random.randint(60, 95)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            result, encimg = cv2.imencode('.jpg', image, encode_param)
            image = cv2.imdecode(encimg, 1)

        return image

    def place_offers(self, offers, min_offers=2, max_offers=8):
        """Place offers on the canvas including challenging cases with COCO annotations"""
        # Create background
        canvas = self.create_complex_background()

        # Select offers
        num_offers = random.randint(min_offers, min(max_offers, len(offers)))
        selected_offers = random.sample(offers, num_offers)

        annotations = []
        used_positions = []

        # Choose placement strategy
        scenarios = ['grid', 'adjacent', 'matching_bg']
        scenario = random.choice(scenarios)

        if scenario == 'grid':
            # Original grid-based placement
            rows = int(np.ceil(np.sqrt(num_offers)))
            cols = int(np.ceil(num_offers / rows))
            cell_width = self.width // cols
            cell_height = self.height // rows

            for i, offer_path in enumerate(selected_offers):
                try:
                    offer_img = cv2.imread(str(offer_path))
                    if offer_img is None:
                        continue

                    # Grid position calculation and placement
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

                    offer_img = cv2.resize(offer_img, (new_width, new_height))
                    x = col * cell_width + (cell_width - new_width) // 2
                    y = row * cell_height + (cell_height - new_height) // 2

                    x += random.randint(-20, 20)
                    y += random.randint(-20, 20)

                    canvas, annotation = self._place_single_offer(
                        canvas, offer_img, x, y, len(self.coco_dataset["images"]) + 1
                    )
                    if annotation:
                        annotations.append(annotation)
                        used_positions.append(annotation["bbox"])

                except Exception as e:
                    print(f"Error processing offer {offer_path}: {str(e)}")
                    continue

        elif scenario == 'adjacent':
            # Place offers adjacent to each other
            try:
                total_width = 0
                offer_images = []

                # First pass: load and resize images
                for offer_path in selected_offers:
                    offer_img = cv2.imread(str(offer_path))
                    if offer_img is None:
                        continue

                    # Use consistent height for adjacent offers
                    target_height = int(self.height * 0.3)  # 30% of canvas height
                    aspect_ratio = offer_img.shape[1] / offer_img.shape[0]
                    new_width = int(target_height * aspect_ratio)

                    offer_img = cv2.resize(offer_img, (new_width, target_height))
                    total_width += new_width
                    offer_images.append(offer_img)

                # Calculate starting position
                start_x = random.randint(0, max(0, self.width - total_width))
                y = random.randint(0, self.height - target_height)

                current_x = start_x
                for offer_img in offer_images:
                    canvas, annotation = self._place_single_offer(
                        canvas, offer_img, current_x, y, len(self.coco_dataset["images"]) + 1
                    )
                    if annotation:
                        annotations.append(annotation)
                        used_positions.append(annotation["bbox"])
                    current_x += offer_img.shape[1]

            except Exception as e:
                print(f"Error in adjacent placement: {str(e)}")

        elif scenario == 'matching_bg':
            # Place offers with matching backgrounds
            for offer_path in selected_offers:
                try:
                    offer_img = cv2.imread(str(offer_path))
                    if offer_img is None:
                        continue

                    # Scale offer
                    target_height = int(self.height * 0.3)
                    aspect_ratio = offer_img.shape[1] / offer_img.shape[0]
                    new_width = int(target_height * aspect_ratio)

                    offer_img = cv2.resize(offer_img, (new_width, target_height))

                    # Random position
                    x = random.randint(0, self.width - new_width)
                    y = random.randint(0, self.height - target_height)

                    # Match background color
                    bg_color = np.mean(canvas[y:y + target_height, x:x + new_width], axis=(0, 1))

                    # Create a mask for the offer (you'll need to adjust this based on your offers)
                    # This is a simple example - you might need more sophisticated masking
                    gray = cv2.cvtColor(offer_img, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
                    mask = cv2.bitwise_not(mask)

                    # Apply background color
                    for c in range(3):
                        offer_img[:, :, c][mask == 0] = bg_color[c]

                    canvas, annotation = self._place_single_offer(
                        canvas, offer_img, x, y, len(self.coco_dataset["images"]) + 1
                    )
                    if annotation:
                        annotations.append(annotation)
                        used_positions.append(annotation["bbox"])

                except Exception as e:
                    print(f"Error in matching_bg placement: {str(e)}")

        return self.apply_augmentation(canvas), annotations

    def _place_single_offer(self, canvas, offer_img, x, y, image_id):
        """Helper method to place a single offer and create its annotation"""
        try:
            # Ensure coordinates are within bounds
            x = max(0, min(x, self.width - offer_img.shape[1]))
            y = max(0, min(y, self.height - offer_img.shape[0]))

            # Get region of interest dimensions
            roi_height = min(offer_img.shape[0], self.height - y)
            roi_width = min(offer_img.shape[1], self.width - x)

            # Ensure offer image is cropped to match ROI if needed
            offer_roi = offer_img[:roi_height, :roi_width]

            # Place offer
            canvas[y:y + roi_height, x:x + roi_width] = offer_roi

            # Create annotation
            annotation = {
                "id": self.annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x, y, roi_width, roi_height],
                "area": roi_width * roi_height,
                "segmentation": [],
                "iscrowd": 0
            }
            self.annotation_id += 1

            return canvas, annotation

        except Exception as e:
            print(f"Error in _place_single_offer: {str(e)}")
            return canvas, None

    @staticmethod
    def check_overlap(bbox1, bbox2, threshold=0):
        """Check if two bounding boxes overlap"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        return not (x1_max < x2_min - threshold or
                    x1_min > x2_max + threshold or
                    y1_max < y2_min - threshold or
                    y1_min > y2_max + threshold)

    def generate_dataset(self, num_layouts=100):
        """Generate dataset with dimension-safe image handling"""
        # Get all offer images
        offer_paths = []
        for ext in ['.jpg', '.jpeg', '.png']:
            offer_paths.extend(Path(self.input_folder).rglob(f'*{ext}'))

        if not offer_paths:
            raise ValueError(f"No images found in {self.input_folder}")

        print(f"Found {len(offer_paths)} offer images")
        for i in range(num_layouts):
            try:
                # Generate layout
                canvas, annotations = self.place_offers(offer_paths)

                if len(annotations) > 0:
                    # Save image
                    image_filename = f'ad_{i + self.start_with:06d}.jpg'
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


def generate_offer_dataset(input_folder, non_offer_folder, output_dir='ad_dataset_2', num_layouts=100, start_with=1):
    """Convenience function to generate offer detection dataset"""
    generator = EnhancedOfferLayoutGenerator(input_folder, non_offer_folder, output_dir, start_with)

    print("Starting dataset generation...")
    print(f"Input folder: {input_folder}")
    print(f"Non offer folder: {non_offer_folder}")
    print(f"Output directory: {output_dir}")
    print(f"Number of layouts to generate: {num_layouts}")
    print(f"When naming output images start with: {start_with}")

    generator.generate_dataset(num_layouts)




if __name__ == "__main__":
    # Example usage
    input_folder = "/Users/steene/PycharmProjects/RekognitionExperiment/mt-input3"
    non_offer_folder = "/Users/steene/PycharmProjects/RekognitionExperiment/synthetic/product_images"
    generate_offer_dataset(input_folder, non_offer_folder, output_dir="ad_dataset_3", num_layouts=11, start_with=1)
