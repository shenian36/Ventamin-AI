"""
Ventamin AI Video Generator
Specialized video generator for Ventamin Light Up and Clear It products
"""

import json
import time
import logging
from datetime import datetime
from pathlib import Path
import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import numpy as np
import io
import wave
import struct

# Try to import free AI libraries
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from moviepy import ImageSequenceClip, VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip, ImageClip, ColorClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

from config import GENERATED_ADS_DIR, VENTAMIN_BRAND
from src.utils.dataset_loader import VentaminDatasetLoader

class VentaminAIVideoGenerator:
    def __init__(self):
        self.setup_logging()
        self.dataset_loader = VentaminDatasetLoader()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load available images from dataset
        self.load_dataset_images()
        
        # Ventamin product information
        self.products = {
            "light_up": {
                "name": "Ventamin Light Up",
                "headlines": [
                    "From Within Radiance",
                    "Glow Up Naturally",
                    "Illuminate Your Skin",
                    "Radiance from Within"
                ],
                "emotional_hooks": [
                    "Tired of dull, lackluster skin?",
                    "Feel frustrated by persistent dark spots?",
                    "Want that natural glow without heavy routines?",
                    "Struggling with uneven skin tone?"
                ],
                "benefits": [
                    "Fades pigmentation naturally",
                    "Boosts skin hydration",
                    "Improves skin texture",
                    "Protects from UV damage",
                    "Enhances natural radiance"
                ],
                "ingredients": [
                    "Fern Extract - UV protection",
                    "N-acetylcysteine - fights melasma",
                    "Probiotics - regulate skin immunity",
                    "Natural antioxidants"
                ],
                "usage": "Once daily mix-with-water sachet, optional second dose before long sun exposure",
                "target_audience": "Adults wanting a glow-up and clearer skin without heavy topical routines"
            },
            "clear_it": {
                "name": "Ventamin Clear It",
                "headlines": [
                    "Detox & Clear with One Sachet",
                    "Clear Skin from Within",
                    "Break Free from Breakouts",
                    "Pure Skin, Naturally"
                ],
                "emotional_hooks": [
                    "Tired of recurring breakouts?",
                    "Frustrated by persistent acne?",
                    "Struggling with acne-prone skin?",
                    "Want clear skin without harsh treatments?"
                ],
                "benefits": [
                    "Reduces acne breakouts",
                    "Balances sebum production",
                    "Detoxifies skin naturally",
                    "Improves skin clarity",
                    "Teen-friendly formula"
                ],
                "ingredients": [
                    "Natural detoxifiers",
                    "Sebum-balancing compounds",
                    "Anti-inflammatory agents",
                    "Skin-clarifying nutrients"
                ],
                "usage": "Once daily mix-with-water sachet, teen-friendly formula",
                "target_audience": "Young adults aged 18-30 struggling with pigmentation or acne-prone, dull skin"
            }
        }
        
        # Social proof and CTA
        self.social_proof = "94% of users reported visible skin improvement in 12 weeks"
        self.cta_options = [
            "Shop Now",
            "Get a 90-day money-back guarantee",
            "Free shipping globally",
            "Start Your Transformation Today"
        ]
        
        # Audio generation settings
        self.sample_rate = 44100
        self.audio_segments = []
    
    def load_dataset_images(self):
        """Load available images from the dataset"""
        self.available_images = {
            "products": [],
            "lifestyle": [],
            "testimonials": [],
            "branding": [],
            "stock_footage": []
        }
        
        try:
            # Load product images
            product_images = self.dataset_loader.get_available_assets("products")
            self.available_images["products"] = [img for img in product_images if img.endswith(('.jpg', '.jpeg', '.png', '.avif'))]
            
            # Load lifestyle images
            lifestyle_images = self.dataset_loader.get_available_assets("lifestyle")
            self.available_images["lifestyle"] = [img for img in lifestyle_images if img.endswith(('.jpg', '.jpeg', '.png', '.avif'))]
            
            # Load testimonial images
            testimonial_images = self.dataset_loader.get_available_assets("testimonials")
            self.available_images["testimonials"] = [img for img in testimonial_images if img.endswith(('.jpg', '.jpeg', '.png', '.avif'))]
            
            # Load branding images
            branding_images = self.dataset_loader.get_available_assets("branding")
            self.available_images["branding"] = [img for img in branding_images if img.endswith(('.jpg', '.jpeg', '.png', '.avif'))]
            
            # Load stock footage images
            stock_images = self.dataset_loader.get_available_assets("stock_footage")
            self.available_images["stock_footage"] = [img for img in stock_images if img.endswith(('.jpg', '.jpeg', '.png', '.avif'))]
            
            self.logger.info(f"Loaded {len(self.available_images['products'])} product images")
            self.logger.info(f"Loaded {len(self.available_images['lifestyle'])} lifestyle images")
            self.logger.info(f"Loaded {len(self.available_images['testimonials'])} testimonial images")
            self.logger.info(f"Loaded {len(self.available_images['branding'])} branding images")
            self.logger.info(f"Loaded {len(self.available_images['stock_footage'])} stock footage images")
            
        except Exception as e:
            self.logger.warning(f"Error loading dataset images: {e}")
            self.available_images = {"products": [], "lifestyle": [], "testimonials": [], "branding": [], "stock_footage": []}
    
    def get_random_image(self, category, size=(400, 400)):
        """Get a random image from the specified category and resize it"""
        if not self.available_images[category]:
            return self.create_generated_background(size, category)
        
        try:
            image_path = random.choice(self.available_images[category])
            image = Image.open(image_path)
            
            # Resize and maintain aspect ratio
            image.thumbnail(size, Image.Resampling.LANCZOS)
            
            # Create a new image with the target size and paste the resized image
            new_image = Image.new('RGBA', size, (255, 255, 255, 0))
            
            # Calculate position to center the image
            x = (size[0] - image.width) // 2
            y = (size[1] - image.height) // 2
            
            new_image.paste(image, (x, y))
            return new_image
            
        except Exception as e:
            self.logger.warning(f"Error loading image {image_path}: {e}")
            return self.create_generated_background(size, category)
    
    def create_generated_background(self, size, category):
        """Create a generated background when no images are available"""
        width, height = size
        
        # Create different backgrounds based on category
        if category == "products":
            # Create a product-like background with gradient
            image = Image.new('RGB', (width, height))
            draw = ImageDraw.Draw(image)
            
            # Create gradient background
            for y in range(height):
                r = int(74 + (y / height) * 50)
                g = int(144 + (y / height) * 30)
                b = int(226 + (y / height) * 20)
                draw.line([(0, y), (width, y)], fill=(r, g, b))
            
            # Add some product-like elements
            center_x, center_y = width // 2, height // 2
            draw.ellipse([center_x-50, center_y-50, center_x+50, center_y+50], 
                        fill=(255, 255, 255, 100), outline=(255, 255, 255, 200))
            
        elif category == "lifestyle":
            # Create a lifestyle background
            image = Image.new('RGB', (width, height), (240, 248, 255))
            draw = ImageDraw.Draw(image)
            
            # Add some lifestyle elements
            for i in range(5):
                x = random.randint(0, width)
                y = random.randint(0, height)
                size_circle = random.randint(20, 60)
                draw.ellipse([x-size_circle, y-size_circle, x+size_circle, y+size_circle], 
                           fill=(255, 255, 255, 50))
            
        elif category == "testimonials":
            # Create a testimonial background
            image = Image.new('RGB', (width, height), (255, 248, 240))
            draw = ImageDraw.Draw(image)
            
            # Add testimonial-like elements
            center_x, center_y = width // 2, height // 2
            draw.rectangle([center_x-80, center_y-40, center_x+80, center_y+40], 
                         fill=(255, 255, 255, 100), outline=(74, 144, 226))
            
        elif category == "branding":
            # Create a branding background
            image = Image.new('RGB', (width, height), (74, 144, 226))
            draw = ImageDraw.Draw(image)
            
            # Add branding elements
            center_x, center_y = width // 2, height // 2
            draw.text((center_x-30, center_y-10), "VENTAMIN", fill=(255, 255, 255))
            
        else:
            # Default background
            image = Image.new('RGB', (width, height), (255, 255, 255))
            draw = ImageDraw.Draw(image)
            
            # Add some decorative elements
            for i in range(3):
                x = random.randint(0, width)
                y = random.randint(0, height)
                draw.rectangle([x-20, y-20, x+20, y+20], 
                             fill=(74, 144, 226, 50))
        
        return image
    
    def create_image_overlay(self, base_image, overlay_image, position='center', opacity=0.8):
        """Create an overlay of an image on the base image"""
        if overlay_image is None:
            return base_image
        
        # Convert base image to RGBA if needed
        if base_image.mode != 'RGBA':
            base_image = base_image.convert('RGBA')
        
        # Apply opacity to overlay
        overlay_image.putalpha(int(255 * opacity))
        
        # Calculate position
        if position == 'center':
            x = (base_image.width - overlay_image.width) // 2
            y = (base_image.height - overlay_image.height) // 2
        elif position == 'top':
            x = (base_image.width - overlay_image.width) // 2
            y = 50
        elif position == 'bottom':
            x = (base_image.width - overlay_image.width) // 2
            y = base_image.height - overlay_image.height - 50
        else:
            x, y = position
        
        # Create a copy of the base image
        result = base_image.copy()
        result.paste(overlay_image, (x, y), overlay_image)
        
        return result
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def create_ventamin_frames(self, product_type, duration=15, fps=30):
        """Create Ventamin-specific animated frames with images"""
        frames = []
        total_frames = duration * fps
        
        if product_type == "light_up":
            scenes = self._create_light_up_scenes(total_frames)
        elif product_type == "clear_it":
            scenes = self._create_clear_it_scenes(total_frames)
        else:
            scenes = self._create_bundle_scenes(total_frames)
        
        return scenes
    
    def _create_light_up_scenes(self, total_frames):
        """Create Light Up specific scenes with images"""
        scenes = []
        frames_per_scene = total_frames // 6
        
        product_info = self.products["light_up"]
        
        # Pre-select images for each scene to prevent flashing
        scene1_image = self.get_random_image("lifestyle", (1080, 1920))
        scene2_image = self.get_random_image("products", (500, 500))
        scene3_image = self.get_random_image("testimonials", (400, 400))
        scene4_image = self.get_random_image("products", (350, 350))
        scene5_image = self.get_random_image("branding", (200, 200))
        scene6_image = self.get_random_image("products", (450, 450))
        
        # Scene 1: Emotional Hook with lifestyle image
        for i in range(frames_per_scene):
            frame = self._create_emotional_hook_frame_with_image(i, frames_per_scene, product_info["emotional_hooks"][0], "lifestyle", scene1_image)
            scenes.append(frame)
        
        # Scene 2: Headline with product image
        for i in range(frames_per_scene):
            frame = self._create_headline_frame_with_image(i, frames_per_scene, product_info["headlines"][0], "products", scene2_image)
            scenes.append(frame)
        
        # Scene 3: Benefits with testimonial image
        for i in range(frames_per_scene):
            frame = self._create_benefits_frame_with_image(i, frames_per_scene, product_info["benefits"], "testimonials", scene3_image)
            scenes.append(frame)
        
        # Scene 4: Ingredients with product image
        for i in range(frames_per_scene):
            frame = self._create_ingredients_frame_with_image(i, frames_per_scene, product_info["ingredients"], "products", scene4_image)
            scenes.append(frame)
        
        # Scene 5: Social Proof with branding image
        for i in range(frames_per_scene):
            frame = self._create_social_proof_frame_with_image(i, frames_per_scene, "branding", scene5_image)
            scenes.append(frame)
        
        # Scene 6: CTA with product image
        for i in range(frames_per_scene):
            frame = self._create_cta_frame_with_image(i, frames_per_scene, self.cta_options[0], "products", scene6_image)
            scenes.append(frame)
        
        return scenes
    
    def _create_clear_it_scenes(self, total_frames):
        """Create Clear It specific scenes with images"""
        scenes = []
        frames_per_scene = total_frames // 6
        
        product_info = self.products["clear_it"]
        
        # Pre-select images for each scene to prevent flashing
        scene1_image = self.get_random_image("lifestyle", (1080, 1920))
        scene2_image = self.get_random_image("products", (500, 500))
        scene3_image = self.get_random_image("testimonials", (400, 400))
        scene4_image = self.get_random_image("products", (350, 350))
        scene5_image = self.get_random_image("branding", (200, 200))
        scene6_image = self.get_random_image("products", (450, 450))
        
        # Scene 1: Emotional Hook with lifestyle image
        for i in range(frames_per_scene):
            frame = self._create_emotional_hook_frame_with_image(i, frames_per_scene, product_info["emotional_hooks"][0], "lifestyle", scene1_image)
            scenes.append(frame)
        
        # Scene 2: Headline with product image
        for i in range(frames_per_scene):
            frame = self._create_headline_frame_with_image(i, frames_per_scene, product_info["headlines"][0], "products", scene2_image)
            scenes.append(frame)
        
        # Scene 3: Benefits with testimonial image
        for i in range(frames_per_scene):
            frame = self._create_benefits_frame_with_image(i, frames_per_scene, product_info["benefits"], "testimonials", scene3_image)
            scenes.append(frame)
        
        # Scene 4: Ingredients with product image
        for i in range(frames_per_scene):
            frame = self._create_ingredients_frame_with_image(i, frames_per_scene, product_info["ingredients"], "products", scene4_image)
            scenes.append(frame)
        
        # Scene 5: Social Proof with branding image
        for i in range(frames_per_scene):
            frame = self._create_social_proof_frame_with_image(i, frames_per_scene, "branding", scene5_image)
            scenes.append(frame)
        
        # Scene 6: CTA with product image
        for i in range(frames_per_scene):
            frame = self._create_cta_frame_with_image(i, frames_per_scene, self.cta_options[0], "products", scene6_image)
            scenes.append(frame)
        
        return scenes
    
    def _create_bundle_scenes(self, total_frames):
        """Create bundle scenes featuring both products with images"""
        scenes = []
        frames_per_scene = total_frames // 7
        
        # Scene 1: Emotional Hook (Universal) with lifestyle image
        for i in range(frames_per_scene):
            frame = self._create_emotional_hook_frame_with_image(i, frames_per_scene, "Want clear, radiant skin naturally?", "lifestyle")
            scenes.append(frame)
        
        # Scene 2: Light Up Introduction with product image
        for i in range(frames_per_scene):
            frame = self._create_product_intro_frame_with_image(i, frames_per_scene, "Light Up", "For Radiance", "products")
            scenes.append(frame)
        
        # Scene 3: Clear It Introduction with product image
        for i in range(frames_per_scene):
            frame = self._create_product_intro_frame_with_image(i, frames_per_scene, "Clear It", "For Clarity", "products")
            scenes.append(frame)
        
        # Scene 4: Combined Benefits with testimonial image
        for i in range(frames_per_scene):
            frame = self._create_combined_benefits_frame_with_image(i, frames_per_scene, "testimonials")
            scenes.append(frame)
        
        # Scene 5: Bundle Value with product images
        for i in range(frames_per_scene):
            frame = self._create_bundle_value_frame_with_image(i, frames_per_scene, "products")
            scenes.append(frame)
        
        # Scene 6: Social Proof with branding image
        for i in range(frames_per_scene):
            frame = self._create_social_proof_frame_with_image(i, frames_per_scene, "branding")
            scenes.append(frame)
        
        # Scene 7: CTA with product image
        for i in range(frames_per_scene):
            frame = self._create_cta_frame_with_image(i, frames_per_scene, "Get Both Products", "products")
            scenes.append(frame)
        
        return scenes
    
    def _create_emotional_hook_frame_with_image(self, frame_num, total_frames, hook_text, image_category, pre_selected_image=None):
        """Create emotional hook frame with background image"""
        width, height = 1080, 1920
        
        # Use pre-selected image or get a new one
        if pre_selected_image:
            bg_image = pre_selected_image
        else:
            bg_image = self.get_random_image(image_category, (width, height))
        
        if bg_image:
            image = bg_image.convert('RGB')
        else:
            image = Image.new('RGB', (width, height), color=(74, 144, 226))
        
        draw = ImageDraw.Draw(image)
        
        # Add a semi-transparent overlay for better text readability
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 80))
        image = self.create_image_overlay(image, overlay, opacity=0.6)
        draw = ImageDraw.Draw(image)
        
        # Add emotional hook text - make it appear quickly and stay clear
        text_opacity = min(255, int(255 * (frame_num / (total_frames * 0.2))))  # Appear in first 20% of scene
        try:
            font = ImageFont.truetype("arial.ttf", 56)  # Larger font
        except:
            font = ImageFont.load_default()
        
        # Split text for better layout
        words = hook_text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] < width - 100:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        # Draw lines with strong shadow for better readability
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            x = (width - text_width) // 2
            y = height // 3 + (i * 90)
            
            # Draw strong shadow
            draw.text((x+3, y+3), line, font=font, fill=(0, 0, 0, text_opacity))
            # Draw main text
            draw.text((x, y), line, font=font, fill=(255, 255, 255, text_opacity))
        
        return image
    
    def _create_headline_frame_with_image(self, frame_num, total_frames, headline, image_category, pre_selected_image=None):
        """Create headline frame with product image"""
        width, height = 1080, 1920
        image = Image.new('RGB', (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # Add brand color accent
        draw.rectangle([0, 0, width, 100], fill=(74, 144, 226))
        
        # Use pre-selected image or get a new one
        if pre_selected_image:
            product_image = pre_selected_image
        else:
            product_image = self.get_random_image(image_category, (500, 500))  # Larger image
        
        if product_image:
            # Position image in center-right area
            x = width - 550
            y = height // 2 - 250
            image = self.create_image_overlay(image, product_image, position=(x, y), opacity=1.0)  # Full opacity
            draw = ImageDraw.Draw(image)
        
        # Add headline text - appear quickly and stay clear
        text_opacity = min(255, int(255 * (frame_num / (total_frames * 0.3))))  # Appear in first 30% of scene
        try:
            font = ImageFont.truetype("arial.ttf", 64)  # Larger font
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), headline, font=font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        y = height // 2 + 100  # Position below image
        
        # Add shadow for better readability
        draw.text((x+2, y+2), headline, font=font, fill=(0, 0, 0, text_opacity))
        draw.text((x, y), headline, font=font, fill=(74, 144, 226, text_opacity))
        
        return image
    
    def _create_benefits_frame_with_image(self, frame_num, total_frames, benefits, image_category, pre_selected_image=None):
        """Create benefits frame with testimonial image"""
        width, height = 1080, 1920
        image = Image.new('RGB', (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # Use pre-selected image or get a new one
        if pre_selected_image:
            testimonial_image = pre_selected_image
        else:
            testimonial_image = self.get_random_image(image_category, (400, 400))  # Larger image
        
        if testimonial_image:
            # Position image in center-left area
            x = 80
            y = height // 2 - 200
            image = self.create_image_overlay(image, testimonial_image, position=(x, y), opacity=1.0)  # Full opacity
            draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 40)  # Larger font
        except:
            font = ImageFont.load_default()
        
        # Add benefits title - appear quickly
        title_opacity = min(255, int(255 * (frame_num / (total_frames * 0.2))))
        title = "Key Benefits"
        bbox = draw.textbbox((0, 0), title, font=font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        y = height // 6
        draw.text((x+2, y+2), title, font=font, fill=(0, 0, 0, title_opacity))
        draw.text((x, y), title, font=font, fill=(74, 144, 226, title_opacity))
        
        # Add benefits list (positioned on the right side) - appear quickly
        for i, benefit in enumerate(benefits[:4]):  # Show first 4 benefits
            benefit_opacity = min(255, int(255 * (frame_num / (total_frames * 0.4))))
            y_pos = height // 3 + (i * 90)
            draw.text((500, y_pos), f"✓ {benefit}", font=font, fill=(0, 0, 0, benefit_opacity))
        
        return image
    
    def _create_ingredients_frame_with_image(self, frame_num, total_frames, ingredients, image_category, pre_selected_image=None):
        """Create ingredients frame with product image"""
        width, height = 1080, 1920
        image = Image.new('RGB', (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # Use pre-selected image or get a new one
        if pre_selected_image:
            product_image = pre_selected_image
        else:
            product_image = self.get_random_image(image_category, (350, 350))
        
        if product_image:
            # Position image on the right side
            x = width - 400
            y = height // 2 - 175
            image = self.create_image_overlay(image, product_image, position=(x, y), opacity=0.9)
            draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 32)
        except:
            font = ImageFont.load_default()
        
        # Add ingredients title
        title = "Core Ingredients"
        bbox = draw.textbbox((0, 0), title, font=font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        y = height // 6
        draw.text((x, y), title, font=font, fill=(74, 144, 226))
        
        # Add ingredients list (positioned on the left side)
        for i, ingredient in enumerate(ingredients[:4]):  # Show first 4 ingredients
            if frame_num >= (i * total_frames // len(ingredients)):
                y_pos = height // 3 + (i * 70)
                draw.text((100, y_pos), f"• {ingredient}", font=font, fill=(0, 0, 0))
        
        return image
    
    def _create_social_proof_frame_with_image(self, frame_num, total_frames, image_category, pre_selected_image=None):
        """Create social proof frame with branding image"""
        width, height = 1080, 1920
        image = Image.new('RGB', (width, height), color=(74, 144, 226))
        draw = ImageDraw.Draw(image)
        
        # Use pre-selected image or get a new one
        if pre_selected_image:
            branding_image = pre_selected_image
        else:
            branding_image = self.get_random_image(image_category, (200, 200))
        
        if branding_image:
            # Position image at the top
            x = (width - 200) // 2
            y = 100
            image = self.create_image_overlay(image, branding_image, position=(x, y), opacity=0.8)
            draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 42)
        except:
            font = ImageFont.load_default()
        
        # Add social proof text
        text_opacity = int(255 * (frame_num / total_frames))
        bbox = draw.textbbox((0, 0), self.social_proof, font=font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        y = height // 2
        
        draw.text((x, y), self.social_proof, font=font, fill=(255, 255, 255, text_opacity))
        
        return image
    
    def _create_cta_frame_with_image(self, frame_num, total_frames, cta_text, image_category, pre_selected_image=None):
        """Create call to action frame with product image"""
        width, height = 1080, 1920
        image = Image.new('RGB', (width, height), color=(74, 144, 226))
        draw = ImageDraw.Draw(image)
        
        # Use pre-selected image or get a new one
        if pre_selected_image:
            product_image = pre_selected_image
        else:
            product_image = self.get_random_image(image_category, (450, 450))  # Larger image
        
        if product_image:
            # Position image in center-top area
            x = (width - 450) // 2
            y = 150
            image = self.create_image_overlay(image, product_image, position=(x, y), opacity=1.0)  # Full opacity
            draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 58)  # Larger font
        except:
            font = ImageFont.load_default()
        
        # Add CTA text - appear quickly and stay clear
        text_opacity = min(255, int(255 * (frame_num / (total_frames * 0.3))))
        bbox = draw.textbbox((0, 0), cta_text, font=font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        y = height // 2 + 150
        
        # Add shadow for better readability
        draw.text((x+2, y+2), cta_text, font=font, fill=(0, 0, 0, text_opacity))
        draw.text((x, y), cta_text, font=font, fill=(255, 255, 255, text_opacity))
        
        return image
    
    def _create_product_intro_frame_with_image(self, frame_num, total_frames, product_name, tagline, image_category):
        """Create product introduction frame with product image"""
        width, height = 1080, 1920
        image = Image.new('RGB', (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # Add product image
        product_image = self.get_random_image(image_category, (400, 400))
        if product_image:
            # Position image in the center
            x = (width - 400) // 2
            y = height // 2 - 100
            image = self.create_image_overlay(image, product_image, position=(x, y), opacity=0.9)
            draw = ImageDraw.Draw(image)
        
        try:
            title_font = ImageFont.truetype("arial.ttf", 48)
            subtitle_font = ImageFont.load_default()
        except:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
        
        # Add product name
        bbox = draw.textbbox((0, 0), product_name, font=title_font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        y = height // 2 + 200
        draw.text((x, y), product_name, font=title_font, fill=(74, 144, 226))
        
        # Add tagline
        bbox = draw.textbbox((0, 0), tagline, font=subtitle_font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        y = height // 2 + 250
        draw.text((x, y), tagline, font=subtitle_font, fill=(0, 0, 0))
        
        return image
    
    def _create_combined_benefits_frame_with_image(self, frame_num, total_frames, image_category):
        """Create combined benefits frame for bundle with testimonial image"""
        width, height = 1080, 1920
        image = Image.new('RGB', (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # Add testimonial image
        testimonial_image = self.get_random_image(image_category, (300, 300))
        if testimonial_image:
            # Position image on the left side
            x = 50
            y = height // 2 - 150
            image = self.create_image_overlay(image, testimonial_image, position=(x, y), opacity=0.8)
            draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except:
            font = ImageFont.load_default()
        
        title = "Complete Skin Transformation"
        bbox = draw.textbbox((0, 0), title, font=font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        y = height // 6
        draw.text((x, y), title, font=font, fill=(74, 144, 226))
        
        benefits = [
            "Clear skin from within",
            "Natural radiance boost",
            "Teen-friendly formula",
            "UV protection included"
        ]
        
        for i, benefit in enumerate(benefits):
            if frame_num >= (i * total_frames // len(benefits)):
                y_pos = height // 3 + (i * 80)
                draw.text((400, y_pos), f"✓ {benefit}", font=font, fill=(0, 0, 0))
        
        return image
    
    def _create_bundle_value_frame_with_image(self, frame_num, total_frames, image_category):
        """Create bundle value frame with product images"""
        width, height = 1080, 1920
        image = Image.new('RGB', (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # Add two product images side by side
        product_image1 = self.get_random_image(image_category, (250, 250))
        product_image2 = self.get_random_image(image_category, (250, 250))
        
        if product_image1:
            # Position first product on the left
            x1 = width // 2 - 300
            y1 = height // 2 - 125
            image = self.create_image_overlay(image, product_image1, position=(x1, y1), opacity=0.9)
            draw = ImageDraw.Draw(image)
        
        if product_image2:
            # Position second product on the right
            x2 = width // 2 + 50
            y2 = height // 2 - 125
            image = self.create_image_overlay(image, product_image2, position=(x2, y2), opacity=0.9)
            draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 42)
        except:
            font = ImageFont.load_default()
        
        title = "Best Value Bundle"
        bbox = draw.textbbox((0, 0), title, font=font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        y = height // 2 + 150
        draw.text((x, y), title, font=font, fill=(74, 144, 226))
        
        subtitle = "Light Up + Clear It"
        bbox = draw.textbbox((0, 0), subtitle, font=font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        y = height // 2 + 200
        draw.text((x, y), subtitle, font=font, fill=(0, 0, 0))
        
        return image
    
    def generate_sine_wave(self, frequency, duration, amplitude=0.3):
        """Generate a sine wave for audio"""
        samples = int(self.sample_rate * duration)
        audio_data = []
        
        for i in range(samples):
            sample = amplitude * np.sin(2 * np.pi * frequency * i / self.sample_rate)
            audio_data.append(sample)
        
        return audio_data
    
    def generate_voiceover_audio(self, text, duration, tone="neutral"):
        """Generate voiceover audio for text"""
        # Simple tone-based frequency mapping
        if tone == "emotional":
            base_freq = 220  # A3
        elif tone == "confident":
            base_freq = 330  # E4
        elif tone == "urgent":
            base_freq = 440  # A4
        else:
            base_freq = 280  # C#4
        
        # Generate audio with slight variations
        audio_data = []
        samples = int(self.sample_rate * duration)
        
        for i in range(samples):
            # Add slight frequency modulation for natural sound
            mod_freq = base_freq + 10 * np.sin(2 * np.pi * 0.5 * i / self.sample_rate)
            sample = 0.2 * np.sin(2 * np.pi * mod_freq * i / self.sample_rate)
            audio_data.append(sample)
        
        return audio_data
    
    def create_audio_file(self, audio_data, filename):
        """Create a WAV file from audio data"""
        try:
            with wave.open(filename, 'w') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                
                # Convert to 16-bit integers
                audio_int = [int(sample * 32767) for sample in audio_data]
                audio_bytes = struct.pack('<%dh' % len(audio_int), *audio_int)
                wav_file.writeframes(audio_bytes)
            
            self.logger.info(f"Audio file created: {filename}")
            return filename
        except Exception as e:
            self.logger.error(f"Error creating audio file: {e}")
            return None
    
    def generate_scene_audio(self, scene_type, duration):
        """Generate audio for different scene types"""
        if scene_type == "hook":
            return self.generate_voiceover_audio("Emotional hook", duration, "emotional")
        elif scene_type == "headline":
            return self.generate_voiceover_audio("Product headline", duration, "confident")
        elif scene_type == "benefits":
            return self.generate_voiceover_audio("Key benefits", duration, "neutral")
        elif scene_type == "ingredients":
            return self.generate_voiceover_audio("Core ingredients", duration, "neutral")
        elif scene_type == "social_proof":
            return self.generate_voiceover_audio("Social proof", duration, "confident")
        elif scene_type == "cta":
            return self.generate_voiceover_audio("Call to action", duration, "urgent")
        else:
            return self.generate_voiceover_audio("Scene audio", duration, "neutral")
    
    def create_video_from_frames(self, frames, filename, fps=30):
        """Create video from frames with audio"""
        if not MOVIEPY_AVAILABLE:
            self.logger.warning("MoviePy not available, creating frame sequence")
            return self._create_frame_sequence(frames, filename)
        
        try:
            # Save frames as temporary images
            temp_dir = Path("temp_frames")
            temp_dir.mkdir(exist_ok=True)
            
            frame_paths = []
            for i, frame in enumerate(frames):
                frame_path = temp_dir / f"frame_{i:04d}.png"
                frame.save(frame_path)
                frame_paths.append(str(frame_path))
            
            # Generate audio for the video
            total_duration = len(frames) / fps
            scene_duration = total_duration / 6  # 6 scenes
            
            # Generate audio for each scene type
            audio_segments = []
            scene_types = ["hook", "headline", "benefits", "ingredients", "social_proof", "cta"]
            
            for scene_type in scene_types:
                scene_audio = self.generate_scene_audio(scene_type, scene_duration)
                audio_segments.extend(scene_audio)
            
            # Create audio file
            audio_filename = filename.replace('.mp4', '_audio.wav')
            audio_path = GENERATED_ADS_DIR / audio_filename
            self.create_audio_file(audio_segments, str(audio_path))
            
            # Create video using MoviePy
            from moviepy import ImageSequenceClip, AudioFileClip
            
            clip = ImageSequenceClip(frame_paths, fps=fps)
            
            # Add audio to video
            if audio_path.exists():
                try:
                    audio_clip = AudioFileClip(str(audio_path))
                    clip = clip.set_audio(audio_clip)
                except Exception as e:
                    self.logger.warning(f"Could not add audio to video: {e}")
            
            video_path = GENERATED_ADS_DIR / filename
            clip.write_videofile(str(video_path), fps=fps, codec='libx264', audio_codec='aac')
            
            # Clean up temporary files
            for frame_path in frame_paths:
                os.remove(frame_path)
            temp_dir.rmdir()
            
            # Clean up audio file
            if audio_path.exists():
                os.remove(audio_path)
            
            return str(video_path)
            
        except Exception as e:
            self.logger.error(f"Error creating video: {e}")
            return self._create_frame_sequence(frames, filename)
    
    def _create_frame_sequence(self, frames, filename):
        """Create a frame sequence when video creation fails"""
        try:
            # Create a directory for the frame sequence
            sequence_dir = GENERATED_ADS_DIR / filename.replace('.mp4', '_frames')
            sequence_dir.mkdir(exist_ok=True)
            
            # Save all frames
            for i, frame in enumerate(frames):
                frame_path = sequence_dir / f"frame_{i:04d}.png"
                frame.save(frame_path)
            
            # Create a metadata file
            metadata = {
                "filename": filename,
                "frame_count": len(frames),
                "fps": 30,
                "resolution": "1080x1920",
                "format": "PNG sequence",
                "generated_at": datetime.now().isoformat(),
                "note": "Video creation failed, frames saved as sequence"
            }
            
            metadata_file = sequence_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Frame sequence saved to {sequence_dir}")
            return str(sequence_dir)
            
        except Exception as e:
            self.logger.error(f"Error creating frame sequence: {e}")
            return None
    
    def generate_light_up_video(self):
        """Generate Light Up product video"""
        self.logger.info("🎬 Generating Ventamin Light Up video with images...")
        
        frames = self.create_ventamin_frames("light_up", duration=24, fps=30)  # 6 scenes × 4 seconds each
        return self.create_video_from_frames(frames, f"ventamin_light_up_with_images_{self.timestamp}.mp4")
    
    def generate_clear_it_video(self):
        """Generate Clear It product video"""
        self.logger.info("🎬 Generating Ventamin Clear It video with images...")
        
        frames = self.create_ventamin_frames("clear_it", duration=24, fps=30)  # 6 scenes × 4 seconds each
        return self.create_video_from_frames(frames, f"ventamin_clear_it_with_images_{self.timestamp}.mp4")
    
    def generate_bundle_video(self):
        """Generate bundle video featuring both products"""
        self.logger.info("🎬 Generating Ventamin Bundle video with images...")
        
        frames = self.create_ventamin_frames("bundle", duration=28, fps=30)  # 7 scenes × 4 seconds each
        return self.create_video_from_frames(frames, f"ventamin_bundle_with_images_{self.timestamp}.mp4")
    
    def generate_all_ventamin_videos(self):
        """Generate all Ventamin product videos with images"""
        videos = {}
        
        # Generate Light Up video
        light_up_video = self.generate_light_up_video()
        if light_up_video:
            videos["light_up"] = {
                "filepath": light_up_video,
                "product": "Ventamin Light Up",
                "target_audience": self.products["light_up"]["target_audience"],
                "description": "Radiance-focused advertisement with product images"
            }
        
        # Generate Clear It video
        clear_it_video = self.generate_clear_it_video()
        if clear_it_video:
            videos["clear_it"] = {
                "filepath": clear_it_video,
                "product": "Ventamin Clear It",
                "target_audience": self.products["clear_it"]["target_audience"],
                "description": "Clarity-focused advertisement with product images"
            }
        
        # Generate Bundle video
        bundle_video = self.generate_bundle_video()
        if bundle_video:
            videos["bundle"] = {
                "filepath": bundle_video,
                "product": "Ventamin Bundle",
                "target_audience": "Young adults wanting complete skin transformation",
                "description": "Bundle promotion advertisement with product images"
            }
        
        return videos

def main():
    """Test the Ventamin AI video generator with images"""
    generator = VentaminAIVideoGenerator()
    
    print("🎬 Ventamin AI Video Generator with Images")
    print("=" * 60)
    print("This will generate targeted Ventamin product videos with:")
    print("• Product images from your dataset")
    print("• Lifestyle and testimonial images")
    print("• Branding elements")
    print("• Emotional hooks and headlines")
    print("• Key benefits and ingredients")
    print("• Social proof and CTAs")
    print()
    
    # Generate all videos
    videos = generator.generate_all_ventamin_videos()
    
    print(f"\n✅ Generated {len(videos)} Ventamin videos with images:")
    for video_type, video_info in videos.items():
        print(f"- {video_type}: {video_info['filepath']}")
        print(f"  Product: {video_info['product']}")
        print(f"  Target: {video_info['target_audience']}")
        print(f"  Description: {video_info['description']}")
    
    print("\n🎬 Ventamin video generation with images completed!")

if __name__ == "__main__":
    main() 