import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import random
from datetime import datetime, timedelta
import textwrap
import colorsys


class AdOfferGenerator:
    def __init__(self, output_dir='ad_dataset'):
        self.output_dir = output_dir
        self.font_sizes = {
            'large': 48,
            'medium': 36,
            'small': 24
        }
        self.offer_types = {
            'discount': [
                '{discount}% OFF',
                'SAVE {discount}%',
                'GET {discount}% OFF',
                '{discount}% DISCOUNT'
            ],
            'bogo': [
                'BUY ONE GET ONE FREE',
                'BOGO FREE',
                'BUY 1 GET 1 FREE',
                '2 FOR 1 SPECIAL'
            ],
            'dollar': [
                'SAVE ${amount}',
                '${amount} OFF',
                'GET ${amount} OFF',
                'SAVE UP TO ${amount}'
            ],
            'free_shipping': [
                'FREE SHIPPING',
                'FREE DELIVERY',
                'SHIPPING INCLUDED',
                'NO SHIPPING COST'
            ]
        }
        self.products = [
            'Shoes', 'Shirts', 'Jeans', 'Watches', 'Bags',
            'Electronics', 'Home Decor', 'Furniture', 'Books',
            'Sports Gear'
        ]
        self.create_directories()

    def create_directories(self):
        """Create necessary directories for dataset"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'annotations'), exist_ok=True)

    def generate_random_color(self, exclude_colors=None):
        """Generate random color ensuring good contrast with excluded colors"""
        if exclude_colors is None:
            exclude_colors = []

        while True:
            # Generate color in HSV space for better control
            hue = random.random()
            saturation = random.uniform(0.4, 0.9)
            value = random.uniform(0.4, 1.0)

            # Convert to RGB
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            color = tuple(int(x * 255) for x in rgb)

            # Check contrast with excluded colors
            has_good_contrast = True
            for exc_color in exclude_colors:
                contrast = abs(sum(color) - sum(exc_color))
                if contrast < 250:  # Minimum contrast threshold
                    has_good_contrast = False
                    break

            if has_good_contrast:
                return color

    def generate_expiry_date(self):
        """Generate random future expiry date"""
        current_date = datetime.now()
        days_ahead = random.randint(7, 90)
        future_date = current_date + timedelta(days=days_ahead)
        return future_date.strftime("%m/%d/%Y")

    def generate_offer_text(self, offer_type):
        """Generate offer text based on type"""
        template = random.choice(self.offer_types[offer_type])

        if offer_type == 'discount':
            discount = random.choice([10, 15, 20, 25, 30, 40, 50, 60, 70])
            return template.format(discount=discount)
        elif offer_type == 'dollar':
            amount = random.choice([5, 10, 15, 20, 25, 30, 50, 100])
            return template.format(amount=amount)
        else:
            return template

    def create_single_ad(self, image_id, size=(800, 400)):
        """Create single advertisement image with offer"""
        # Create base image
        img = Image.new('RGB', size, 'white')
        draw = ImageDraw.Draw(img)

        # Select random offer type and generate text
        offer_type = random.choice(list(self.offer_types.keys()))
        offer_text = self.generate_offer_text(offer_type)

        # Generate colors
        bg_color = self.generate_random_color()
        text_color = self.generate_random_color([bg_color])

        # Fill background
        img = Image.new('RGB', size, bg_color)
        draw = ImageDraw.Draw(img)

        # Add main offer text
        font_size = self.font_sizes['large']
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

        # Calculate text position for center alignment
        text_bbox = draw.textbbox((0, 0), offer_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = (size[0] - text_width) // 2
        text_y = (size[1] - text_height) // 2

        # Draw text with bounding box for annotation
        draw.text((text_x, text_y), offer_text, fill=text_color, font=font)

        # Add additional details
        product = random.choice(self.products)
        expiry_date = self.generate_expiry_date()
        details_text = f"On {product} | Expires {expiry_date}"

        # Add details in smaller font
        font_size_small = self.font_sizes['small']
        try:
            font_small = ImageFont.truetype("arial.ttf", font_size_small)
        except:
            font_small = ImageFont.load_default()

        details_bbox = draw.textbbox((0, 0), details_text, font=font_small)
        details_width = details_bbox[2] - details_bbox[0]
        details_x = (size[0] - details_width) // 2
        details_y = text_y + text_height + 20

        draw.text((details_x, details_y), details_text, fill=text_color, font=font_small)

        # Save image
        image_path = os.path.join(self.output_dir, 'images', f'ad_{image_id}.png')
        img.save(image_path)

        # Create annotation
        annotation = {
            'image_id': image_id,
            'offer_type': offer_type,
            'offer_text': offer_text,
            'product': product,
            'expiry_date': expiry_date,
            'text_bbox': [text_x, text_y, text_x + text_width, text_y + text_height]
        }

        # Save annotation
        annotation_path = os.path.join(self.output_dir, 'annotations', f'ad_{image_id}.txt')
        with open(annotation_path, 'w') as f:
            for key, value in annotation.items():
                f.write(f'{key}: {value}\n')

        return img, annotation

    def generate_dataset(self, num_images=1000):
        """Generate complete dataset of ad offers"""
        dataset_info = []

        for i in range(num_images):
            img, annotation = self.create_single_ad(i)
            dataset_info.append(annotation)

            # Print progress
            if (i + 1) % 100 == 0:
                print(f'Generated {i + 1} images')

        # Save dataset summary
        summary_path = os.path.join(self.output_dir, 'dataset_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f'Total images: {num_images}\n')
            f.write('\nOffer type distribution:\n')
            offer_counts = {}
            for ann in dataset_info:
                offer_counts[ann['offer_type']] = offer_counts.get(ann['offer_type'], 0) + 1
            for offer_type, count in offer_counts.items():
                f.write(f'{offer_type}: {count} ({count / num_images * 100:.1f}%)\n')

generator = AdOfferGenerator(output_dir='my_dataset')

# Generate simple geometric shapes dataset
generator.generate_dataset(num_images=10)
