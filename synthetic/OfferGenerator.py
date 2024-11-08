
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
        scenarios = ['grid', 'adjacent', 'matching_bg', 'netto_style']
        weights = [20, 40, 35, 5] # should add up to 100%
        scenario = random.choices(scenarios, weights=weights, k=1)[0]

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
            canvas, new_annotations = self.place_offers_adjacent(
                canvas,
                selected_offers,
                len(self.coco_dataset["images"]) + 1
            )
            annotations.extend(new_annotations)

        elif scenario == 'matching_bg':
            canvas, new_annotations = self.place_offers_matching_bg(
                canvas,
                selected_offers,
                len(self.coco_dataset["images"]) + 1
            )
            annotations.extend(new_annotations)

        elif scenario == 'netto_style':
            canvas, new_annotations = self.generate_netto_style_ad(
                [cv2.imread(str(path)) for path in selected_offers],
                ["Product " + str(i + 1) for i in range(len(selected_offers))]  # Example descriptions
            )
            annotations.extend(new_annotations)

        return self.apply_augmentation(canvas), annotations

    def place_offers_adjacent(self, canvas, offers, image_id):
        """
        Place offers adjacent to each other without significant overlap

        Args:
            canvas: The background image
            offers: List of offer image paths
            image_id: ID for COCO annotations

        Returns:
            tuple: (modified canvas, list of annotations)
        """
        annotations = []

        try:
            # First pass: load and resize images
            offer_images = []
            total_width = 0
            target_height = int(self.height * 0.3)  # 30% of canvas height

            for offer_path in offers:
                offer_img = cv2.imread(str(offer_path))
                if offer_img is None:
                    continue

                # Maintain aspect ratio while scaling
                aspect_ratio = offer_img.shape[1] / offer_img.shape[0]
                new_width = int(target_height * aspect_ratio)

                offer_img = cv2.resize(offer_img, (new_width, target_height))
                offer_images.append(offer_img)
                total_width += new_width

            if not offer_images:
                return canvas, annotations

            # Add small gap between offers (e.g., 10 pixels)
            gap = 0
            total_width += gap * (len(offer_images) - 1)

            # If total width exceeds canvas width, reduce scale
            # margin = random.uniform(0.95, 1.0)
            margin = 1-(0.1/random.paretovariate(0.2)) # get numbers between 0.90 - 1.0 with more values close to 1.0
            if total_width > self.width:
                scale_factor = (self.width * margin) / total_width  # Leave approx. 1% margin
                total_width = 0
                scaled_images = []

                for img in offer_images:
                    new_width = int(img.shape[1] * scale_factor)
                    new_height = int(img.shape[0] * scale_factor)
                    scaled_img = cv2.resize(img, (new_width, new_height))
                    scaled_images.append(scaled_img)
                    total_width += new_width

                offer_images = scaled_images
                target_height = int(target_height * scale_factor)
                total_width += gap * (len(offer_images) - 1)

            # Calculate starting position to center the group
            start_x = (self.width - total_width) // 2
            # Random vertical position with margin
            y = random.randint(0, self.height - target_height)

            # Place offers
            current_x = start_x
            for offer_img in offer_images:
                # Ensure we're within canvas bounds
                if current_x + offer_img.shape[1] > self.width:
                    break

                # Place offer
                h, w = offer_img.shape[:2]
                canvas[y:y + h, current_x:current_x + w] = offer_img

                # Create annotation
                annotation = {
                    "id": self.annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [current_x, y, w, h],
                    "area": w * h,
                    "segmentation": [],
                    "iscrowd": 0
                }
                annotations.append(annotation)
                self.annotation_id += 1

                # Move to next position (including gap)
                current_x += w + gap

        except Exception as e:
            print(f"Error in adjacent placement: {str(e)}")

        return canvas, annotations

    def place_offers_matching_bg(self, canvas, offers, image_id):
        """
        Place offers with backgrounds matching the canvas, avoiding significant overlap

        Args:
            canvas: The background image
            offers: List of offer image paths
            image_id: ID for COCO annotations

        Returns:
            tuple: (modified canvas, list of annotations)
        """
        annotations = []
        used_positions = []

        for offer_path in offers:
            try:
                # Load and check offer image
                offer_img = cv2.imread(str(offer_path))
                if offer_img is None:
                    continue

                # Scale offer
                percentage_of_canvas_height = random.uniform(0.15, 0.3)
                target_height = int(self.height * percentage_of_canvas_height)  # 10-30% of canvas height
                aspect_ratio = offer_img.shape[1] / offer_img.shape[0]
                new_width = int(target_height * aspect_ratio)

                offer_img = cv2.resize(offer_img, (new_width, target_height))

                # Find position without significant overlap
                position_found = False
                attempts = 0
                while not position_found and attempts < 20:
                    x = random.randint(0, self.width - new_width)
                    y = random.randint(0, self.height - target_height)

                    # Check overlap with existing offers
                    bbox = [x, y, x + new_width, y + target_height]
                    overlap = False
                    for used_bbox in used_positions:
                        if self.check_overlap(bbox, used_bbox, threshold=0.01):
                            overlap = True
                            break

                    if not overlap:
                        position_found = True
                    attempts += 1

                if not position_found:
                    continue

                # Get background region
                bg_region = canvas[y:y + target_height, x:x + new_width].copy()

                # Convert offer to HSV for better color matching
                offer_hsv = cv2.cvtColor(offer_img, cv2.COLOR_BGR2HSV)
                bg_hsv = cv2.cvtColor(bg_region, cv2.COLOR_BGR2HSV)

                # Create mask for offer content
                # Combine multiple techniques for better masking
                masks = []

                # Edge detection based mask
                gray = cv2.cvtColor(offer_img, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                edge_mask = cv2.dilate(edges, None, iterations=3)
                masks.append(edge_mask)

                # Color-based mask
                # Detect white/very light backgrounds
                hsv_mask = cv2.inRange(offer_hsv,
                                       (0, 0, 200),  # lower bound for white/light colors
                                       (180, 30, 255))  # upper bound
                masks.append(hsv_mask)

                # Combine masks
                final_mask = np.zeros_like(gray)
                for mask in masks:
                    final_mask = cv2.bitwise_or(final_mask, mask)

                # Dilate mask to ensure coverage
                final_mask = cv2.dilate(final_mask, None, iterations=2)

                # Create inverse mask for background replacement
                bg_mask = cv2.bitwise_not(final_mask)

                # Create matched offer
                matched_offer = offer_img.copy()

                # Replace background in offer with canvas background
                for c in range(3):
                    matched_offer[:, :, c] = (
                            (bg_mask / 255.0) * bg_region[:, :, c] +
                            (final_mask / 255.0) * offer_img[:, :, c]
                    )

                # Blend edges
                kernel = np.ones((3, 3), np.uint8)
                edge_mask = cv2.dilate(final_mask, kernel, iterations=2) - cv2.erode(final_mask, kernel, iterations=2)
                edge_mask = cv2.GaussianBlur(edge_mask.astype(float), (5, 5), 0)

                for c in range(3):
                    blend = (matched_offer[:, :, c] * (1 - edge_mask / 255.0) +
                             bg_region[:, :, c] * (edge_mask / 255.0))
                    matched_offer[:, :, c] = blend

                # Place offer on canvas
                canvas[y:y + target_height, x:x + new_width] = matched_offer

                # Create annotation
                annotation = {
                    "id": self.annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, new_width, target_height],
                    "area": new_width * target_height,
                    "segmentation": [],
                    "iscrowd": 0
                }
                annotations.append(annotation)
                used_positions.append([x, y, x + new_width, y + target_height])
                self.annotation_id += 1

            except Exception as e:
                print(f"Error in matching_bg placement: {str(e)}")
                continue

        return canvas, annotations

    def check_overlap(self, bbox1, bbox2, threshold=0.1):
        """
        Check if two bounding boxes overlap more than the threshold

        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            threshold: Maximum allowed overlap ratio (0-1)

        Returns:
            bool: True if overlap is greater than threshold
        """
        # Calculate intersection
        x_left = max(bbox1[0], bbox2[0])
        x_right = min(bbox1[2], bbox2[2])
        y_top = max(bbox1[1], bbox2[1])
        y_bottom = min(bbox1[3], bbox2[3])

        if x_right < x_left or y_bottom < y_top:
            return False

        intersection = (x_right - x_left) * (y_bottom - y_top)

        # Calculate areas
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        # Calculate overlap ratio
        overlap_ratio = intersection / min(area1, area2)

        return overlap_ratio > threshold

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

    def create_netto_style_background(self):
        """Create yellow background with Netto-style characteristics"""
        # Create yellow background
        yellow_bg = np.full((self.height, self.width, 3), [0, 238, 255], dtype=np.uint8)  # BGR format

        # Add subtle noise/texture
        noise = np.random.normal(0, 3, (self.height, self.width, 3))
        yellow_bg = np.clip(yellow_bg + noise, 0, 255).astype(np.uint8)

        return yellow_bg

    def create_price_burst(self, price="10,-", size=120):
        """Create black starburst with white price text"""
        # Create circular burst
        burst = np.zeros((size, size, 3), dtype=np.uint8)
        center = size // 2

        # Draw black circle
        cv2.circle(burst, (center, center), size // 2, (0, 0, 0), -1)

        # Add points to create burst effect
        num_points = 16
        for i in range(num_points):
            angle = i * (2 * np.pi / num_points)
            point_x = center + int((size // 2 + 10) * np.cos(angle))
            point_y = center + int((size // 2 + 10) * np.sin(angle))
            cv2.line(burst, (center, center), (point_x, point_y), (0, 0, 0), 5)

        # Add price text
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = size / 120  # Scale based on burst size
        thickness = max(2, int(size / 30))

        # Get text size
        text_size = cv2.getTextSize(price, font, font_scale, thickness)[0]
        text_x = (size - text_size[0]) // 2
        text_y = (size + text_size[1]) // 2

        # Draw white text
        cv2.putText(burst, price, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

        return burst

    def create_netto_style_offer(self, product_img, description, price="10,-"):
        """Create a single Netto-style offer with product image, description, and price burst"""
        # Calculate dimensions
        product_height = int(product_img.shape[0] * 0.7)  # Product takes 70% of height
        total_height = int(product_height * 1.4)  # Additional space for text and price
        total_width = max(product_img.shape[1], int(total_height * 0.8))

        # Create offer canvas
        offer = np.full((total_height, total_width, 3), [0, 238, 255], dtype=np.uint8)  # Yellow background

        # Place product image
        product_resized = cv2.resize(product_img, (int(total_width * 0.8), product_height))
        x_offset = (total_width - product_resized.shape[1]) // 2
        y_offset = 0
        offer[y_offset:y_offset + product_height, x_offset:x_offset + product_resized.shape[1]] = product_resized

        # Add description text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = total_width / 400
        thickness = max(1, int(total_width / 200))

        text_size = cv2.getTextSize(description, font, font_scale, thickness)[0]
        text_x = (total_width - text_size[0]) // 2
        text_y = product_height + text_size[1] + 10

        cv2.putText(offer, description, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

        # Add price burst
        burst_size = int(total_width * 0.4)
        price_burst = self.create_price_burst(price, burst_size)

        burst_x = total_width - burst_size - 10
        burst_y = total_height - burst_size - 10

        # Create mask for burst
        burst_gray = cv2.cvtColor(price_burst, cv2.COLOR_BGR2GRAY)
        _, burst_mask = cv2.threshold(burst_gray, 1, 255, cv2.THRESH_BINARY)

        # Place burst using mask
        for c in range(3):
            offer[burst_y:burst_y + burst_size, burst_x:burst_x + burst_size, c] = \
                np.where(burst_mask > 0, price_burst[:, :, c],
                         offer[burst_y:burst_y + burst_size, burst_x:burst_x + burst_size, c])

        return offer

    def generate_netto_style_ad(self, product_images, descriptions):
        """Generate complete Netto-style advertisement with multiple offers"""
        # Create yellow background
        canvas = self.create_netto_style_background()
        annotations = []

        # Calculate grid layout
        num_offers = len(product_images)
        cols = min(3, num_offers)  # Max 3 columns
        rows = (num_offers + cols - 1) // cols

        cell_width = self.width // cols
        cell_height = self.height // rows

        for i, (product_img, desc) in enumerate(zip(product_images, descriptions)):
            # Calculate grid position
            row = i // cols
            col = i % cols

            # Create offer
            offer = self.create_netto_style_offer(product_img, desc)

            # Scale offer to fit cell
            scale = min(
                (cell_width * 0.9) / offer.shape[1],
                (cell_height * 0.9) / offer.shape[0]
            )

            new_width = int(offer.shape[1] * scale)
            new_height = int(offer.shape[0] * scale)
            offer_resized = cv2.resize(offer, (new_width, new_height))

            # Calculate position
            x = col * cell_width + (cell_width - new_width) // 2
            y = row * cell_height + (cell_height - new_height) // 2

            # Place offer
            canvas[y:y + new_height, x:x + new_width] = offer_resized

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
            self.annotation_id += 1

        return canvas, annotations

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


def generate_offer_dataset(input_folder, non_offer_folder, output_dir='XX', num_layouts=100, start_with=1):
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
    generate_offer_dataset(input_folder, non_offer_folder, output_dir="ad_dataset_4", num_layouts=50, start_with=1)
