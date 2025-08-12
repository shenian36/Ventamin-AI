"""
Background Content Generator
Creates background videos and supplementary footage based on uploaded Ventamin assets
"""

import os
import json
import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np
from datetime import datetime

class BackgroundContentGenerator:
    def __init__(self, dataset_path: str = "assets/ventamin_dataset"):
        self.dataset_path = Path(dataset_path)
        self.stock_footage_path = self.dataset_path / "stock_footage"
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def create_product_backgrounds(self):
        """Create background content based on product images"""
        self.logger.info("🎨 Creating product-based backgrounds...")
        
        products_path = self.dataset_path / "products"
        if not products_path.exists():
            self.logger.warning("No products folder found")
            return
        
        # Get all product images
        product_files = list(products_path.glob("*.jpg")) + list(products_path.glob("*.png"))
        
        for product_file in product_files:
            try:
                # Create different background variations
                self._create_blurred_background(product_file, "blurred")
                self._create_gradient_background(product_file, "gradient")
                self._create_texture_background(product_file, "texture")
                self._create_abstract_background(product_file, "abstract")
                
            except Exception as e:
                self.logger.error(f"Error processing {product_file}: {e}")
    
    def _create_blurred_background(self, product_file, suffix):
        """Create blurred background from product image"""
        try:
            with Image.open(product_file) as img:
                # Resize to background size
                bg_size = (1920, 1080)
                img_resized = img.resize(bg_size, Image.Resampling.LANCZOS)
                
                # Apply blur effect
                blurred = img_resized.filter(ImageFilter.GaussianBlur(radius=20))
                
                # Reduce opacity
                blurred.putalpha(128)
                
                # Create background with brand colors
                background = Image.new('RGB', bg_size, (74, 144, 226))  # Ventamin blue
                
                # Composite blurred image over background
                background.paste(blurred, (0, 0), blurred)
                
                # Save
                output_path = self.stock_footage_path / f"bg_blurred_{product_file.stem}_{suffix}.jpg"
                background.save(output_path, quality=95)
                self.logger.info(f"Created blurred background: {output_path}")
                
        except Exception as e:
            self.logger.error(f"Error creating blurred background: {e}")
    
    def _create_gradient_background(self, product_file, suffix):
        """Create gradient background inspired by product colors"""
        try:
            with Image.open(product_file) as img:
                # Get dominant colors from product
                colors = self._extract_dominant_colors(img, 3)
                
                # Create gradient background
                bg_size = (1920, 1080)
                background = Image.new('RGB', bg_size)
                
                # Create gradient from dominant colors
                for y in range(bg_size[1]):
                    ratio = y / bg_size[1]
                    if len(colors) >= 2:
                        r = int(colors[0][0] * (1 - ratio) + colors[1][0] * ratio)
                        g = int(colors[0][1] * (1 - ratio) + colors[1][1] * ratio)
                        b = int(colors[0][2] * (1 - ratio) + colors[1][2] * ratio)
                    else:
                        r, g, b = colors[0] if colors else (74, 144, 226)
                    
                    background.putpixel((0, y), (r, g, b))
                
                # Save
                output_path = self.stock_footage_path / f"bg_gradient_{product_file.stem}_{suffix}.jpg"
                background.save(output_path, quality=95)
                self.logger.info(f"Created gradient background: {output_path}")
                
        except Exception as e:
            self.logger.error(f"Error creating gradient background: {e}")
    
    def _create_texture_background(self, product_file, suffix):
        """Create textured background based on product"""
        try:
            with Image.open(product_file) as img:
                # Create texture pattern
                bg_size = (1920, 1080)
                texture = Image.new('RGB', bg_size, (255, 255, 255))
                draw = ImageDraw.Draw(texture)
                
                # Create subtle pattern
                for i in range(0, bg_size[0], 50):
                    for j in range(0, bg_size[1], 50):
                        color = (240, 240, 240) if (i + j) % 100 == 0 else (255, 255, 255)
                        draw.rectangle([i, j, i + 50, j + 50], fill=color)
                
                # Add brand color overlay
                overlay = Image.new('RGBA', bg_size, (74, 144, 226, 30))
                texture.paste(overlay, (0, 0), overlay)
                
                # Save
                output_path = self.stock_footage_path / f"bg_texture_{product_file.stem}_{suffix}.jpg"
                texture.save(output_path, quality=95)
                self.logger.info(f"Created texture background: {output_path}")
                
        except Exception as e:
            self.logger.error(f"Error creating texture background: {e}")
    
    def _create_abstract_background(self, product_file, suffix):
        """Create abstract background inspired by product"""
        try:
            bg_size = (1920, 1080)
            background = Image.new('RGB', bg_size, (74, 144, 226))
            draw = ImageDraw.Draw(background)
            
            # Create abstract shapes
            for i in range(10):
                x1 = np.random.randint(0, bg_size[0])
                y1 = np.random.randint(0, bg_size[1])
                x2 = np.random.randint(0, bg_size[0])
                y2 = np.random.randint(0, bg_size[1])
                
                color = (255, 255, 255, 50)
                draw.ellipse([x1, y1, x2, y2], fill=color)
            
            # Save
            output_path = self.stock_footage_path / f"bg_abstract_{product_file.stem}_{suffix}.jpg"
            background.save(output_path, quality=95)
            self.logger.info(f"Created abstract background: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating abstract background: {e}")
    
    def _extract_dominant_colors(self, img, num_colors=3):
        """Extract dominant colors from image"""
        try:
            # Resize for faster processing
            img_small = img.resize((100, 100))
            
            # Convert to RGB if needed
            if img_small.mode != 'RGB':
                img_small = img_small.convert('RGB')
            
            # Get colors
            colors = img_small.getcolors(maxcolors=10000)
            if not colors:
                return [(74, 144, 226)]  # Default Ventamin blue
            
            # Sort by frequency and get top colors
            colors.sort(key=lambda x: x[0], reverse=True)
            dominant_colors = [color[1] for color in colors[:num_colors]]
            
            return dominant_colors
            
        except Exception as e:
            self.logger.error(f"Error extracting colors: {e}")
            return [(74, 144, 226)]
    
    def create_lifestyle_backgrounds(self):
        """Create background content based on lifestyle images"""
        self.logger.info("🏃 Creating lifestyle-based backgrounds...")
        
        lifestyle_path = self.dataset_path / "lifestyle"
        if not lifestyle_path.exists():
            self.logger.warning("No lifestyle folder found")
            return
        
        # Get all lifestyle images
        lifestyle_files = list(lifestyle_path.glob("*.jpg")) + list(lifestyle_path.glob("*.png"))
        
        for lifestyle_file in lifestyle_files:
            try:
                # Create lifestyle-inspired backgrounds
                self._create_wellness_background(lifestyle_file)
                self._create_activity_background(lifestyle_file)
                self._create_nature_background(lifestyle_file)
                
            except Exception as e:
                self.logger.error(f"Error processing {lifestyle_file}: {e}")
    
    def _create_wellness_background(self, lifestyle_file):
        """Create wellness-themed background"""
        try:
            with Image.open(lifestyle_file) as img:
                # Create wellness gradient
                bg_size = (1920, 1080)
                background = Image.new('RGB', bg_size)
                
                # Wellness color palette
                colors = [(74, 144, 226), (255, 255, 255), (240, 248, 255)]
                
                for y in range(bg_size[1]):
                    ratio = y / bg_size[1]
                    if ratio < 0.5:
                        r = int(colors[0][0] * (1 - ratio * 2) + colors[1][0] * ratio * 2)
                        g = int(colors[0][1] * (1 - ratio * 2) + colors[1][1] * ratio * 2)
                        b = int(colors[0][2] * (1 - ratio * 2) + colors[1][2] * ratio * 2)
                    else:
                        r = int(colors[1][0] * (1 - (ratio - 0.5) * 2) + colors[2][0] * (ratio - 0.5) * 2)
                        g = int(colors[1][1] * (1 - (ratio - 0.5) * 2) + colors[2][1] * (ratio - 0.5) * 2)
                        b = int(colors[1][2] * (1 - (ratio - 0.5) * 2) + colors[2][2] * (ratio - 0.5) * 2)
                    
                    background.putpixel((0, y), (r, g, b))
                
                # Save
                output_path = self.stock_footage_path / f"bg_wellness_{lifestyle_file.stem}.jpg"
                background.save(output_path, quality=95)
                self.logger.info(f"Created wellness background: {output_path}")
                
        except Exception as e:
            self.logger.error(f"Error creating wellness background: {e}")
    
    def _create_activity_background(self, lifestyle_file):
        """Create activity-themed background"""
        try:
            bg_size = (1920, 1080)
            background = Image.new('RGB', bg_size, (255, 255, 255))
            draw = ImageDraw.Draw(background)
            
            # Create dynamic pattern
            for i in range(0, bg_size[0], 100):
                for j in range(0, bg_size[1], 100):
                    color = (74, 144, 226, 20)
                    draw.rectangle([i, j, i + 100, j + 100], fill=color)
            
            # Save
            output_path = self.stock_footage_path / f"bg_activity_{lifestyle_file.stem}.jpg"
            background.save(output_path, quality=95)
            self.logger.info(f"Created activity background: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating activity background: {e}")
    
    def _create_nature_background(self, lifestyle_file):
        """Create nature-themed background"""
        try:
            bg_size = (1920, 1080)
            background = Image.new('RGB', bg_size)
            
            # Nature-inspired gradient
            for y in range(bg_size[1]):
                ratio = y / bg_size[1]
                r = int(34 * (1 - ratio) + 139 * ratio)
                g = int(139 * (1 - ratio) + 69 * ratio)
                b = int(34 * (1 - ratio) + 19 * ratio)
                background.putpixel((0, y), (r, g, b))
            
            # Save
            output_path = self.stock_footage_path / f"bg_nature_{lifestyle_file.stem}.jpg"
            background.save(output_path, quality=95)
            self.logger.info(f"Created nature background: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating nature background: {e}")
    
    def create_brand_backgrounds(self):
        """Create background content based on brand assets"""
        self.logger.info("🎨 Creating brand-based backgrounds...")
        
        branding_path = self.dataset_path / "branding"
        if not branding_path.exists():
            self.logger.warning("No branding folder found")
            return
        
        # Create brand-inspired backgrounds
        self._create_logo_background()
        self._create_brand_gradient_background()
        self._create_corporate_background()
    
    def _create_logo_background(self):
        """Create background with subtle logo pattern"""
        try:
            bg_size = (1920, 1080)
            background = Image.new('RGB', bg_size, (255, 255, 255))
            draw = ImageDraw.Draw(background)
            
            # Create subtle brand pattern
            for i in range(0, bg_size[0], 200):
                for j in range(0, bg_size[1], 200):
                    color = (74, 144, 226, 10)
                    draw.ellipse([i, j, i + 200, j + 200], fill=color)
            
            # Save
            output_path = self.stock_footage_path / "bg_brand_logo.jpg"
            background.save(output_path, quality=95)
            self.logger.info(f"Created logo background: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating logo background: {e}")
    
    def _create_brand_gradient_background(self):
        """Create gradient background using brand colors"""
        try:
            bg_size = (1920, 1080)
            background = Image.new('RGB', bg_size)
            
            # Brand color gradient
            brand_colors = [(74, 144, 226), (46, 92, 138), (255, 255, 255)]
            
            for y in range(bg_size[1]):
                ratio = y / bg_size[1]
                if ratio < 0.5:
                    r = int(brand_colors[0][0] * (1 - ratio * 2) + brand_colors[1][0] * ratio * 2)
                    g = int(brand_colors[0][1] * (1 - ratio * 2) + brand_colors[1][1] * ratio * 2)
                    b = int(brand_colors[0][2] * (1 - ratio * 2) + brand_colors[1][2] * ratio * 2)
                else:
                    r = int(brand_colors[1][0] * (1 - (ratio - 0.5) * 2) + brand_colors[2][0] * (ratio - 0.5) * 2)
                    g = int(brand_colors[1][1] * (1 - (ratio - 0.5) * 2) + brand_colors[2][1] * (ratio - 0.5) * 2)
                    b = int(brand_colors[1][2] * (1 - (ratio - 0.5) * 2) + brand_colors[2][2] * (ratio - 0.5) * 2)
                
                background.putpixel((0, y), (r, g, b))
            
            # Save
            output_path = self.stock_footage_path / "bg_brand_gradient.jpg"
            background.save(output_path, quality=95)
            self.logger.info(f"Created brand gradient background: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating brand gradient background: {e}")
    
    def _create_corporate_background(self):
        """Create corporate-style background"""
        try:
            bg_size = (1920, 1080)
            background = Image.new('RGB', bg_size, (255, 255, 255))
            draw = ImageDraw.Draw(background)
            
            # Corporate pattern
            for i in range(0, bg_size[0], 50):
                for j in range(0, bg_size[1], 50):
                    if (i + j) % 100 == 0:
                        color = (240, 240, 240)
                        draw.rectangle([i, j, i + 50, j + 50], fill=color)
            
            # Save
            output_path = self.stock_footage_path / "bg_corporate.jpg"
            background.save(output_path, quality=95)
            self.logger.info(f"Created corporate background: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating corporate background: {e}")
    
    def create_testimonial_backgrounds(self):
        """Create background content based on testimonial images"""
        self.logger.info("💬 Creating testimonial-based backgrounds...")
        
        testimonials_path = self.dataset_path / "testimonials"
        if not testimonials_path.exists():
            self.logger.warning("No testimonials folder found")
            return
        
        # Get all testimonial images
        testimonial_files = list(testimonials_path.glob("*.jpg")) + list(testimonials_path.glob("*.png"))
        
        for testimonial_file in testimonial_files:
            try:
                # Create testimonial-inspired backgrounds
                self._create_success_background(testimonial_file)
                self._create_transformation_background(testimonial_file)
                
            except Exception as e:
                self.logger.error(f"Error processing {testimonial_file}: {e}")
    
    def _create_success_background(self, testimonial_file):
        """Create success-themed background"""
        try:
            bg_size = (1920, 1080)
            background = Image.new('RGB', bg_size)
            
            # Success color gradient
            for y in range(bg_size[1]):
                ratio = y / bg_size[1]
                r = int(255 * (1 - ratio) + 74 * ratio)
                g = int(215 * (1 - ratio) + 144 * ratio)
                b = int(0 * (1 - ratio) + 226 * ratio)
                background.putpixel((0, y), (r, g, b))
            
            # Save
            output_path = self.stock_footage_path / f"bg_success_{testimonial_file.stem}.jpg"
            background.save(output_path, quality=95)
            self.logger.info(f"Created success background: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating success background: {e}")
    
    def _create_transformation_background(self, testimonial_file):
        """Create transformation-themed background"""
        try:
            bg_size = (1920, 1080)
            background = Image.new('RGB', bg_size, (255, 255, 255))
            draw = ImageDraw.Draw(background)
            
            # Transformation pattern
            for i in range(0, bg_size[0], 80):
                for j in range(0, bg_size[1], 80):
                    color = (74, 144, 226, 15)
                    draw.rectangle([i, j, i + 80, j + 80], fill=color)
            
            # Save
            output_path = self.stock_footage_path / f"bg_transformation_{testimonial_file.stem}.jpg"
            background.save(output_path, quality=95)
            self.logger.info(f"Created transformation background: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating transformation background: {e}")
    
    def generate_all_backgrounds(self):
        """Generate all types of background content"""
        self.logger.info("🎬 Starting background content generation...")
        
        # Ensure stock footage directory exists
        self.stock_footage_path.mkdir(exist_ok=True)
        
        # Generate backgrounds based on uploaded content
        self.create_product_backgrounds()
        self.create_lifestyle_backgrounds()
        self.create_brand_backgrounds()
        self.create_testimonial_backgrounds()
        
        self.logger.info("✅ Background content generation completed!")

def main():
    """Test the background content generator"""
    generator = BackgroundContentGenerator()
    
    print("🎨 Ventamin Background Content Generator")
    print("=" * 50)
    print("This will create background content based on your uploaded assets...")
    print()
    
    generator.generate_all_backgrounds()
    
    print("\n✅ Background content generation completed!")
    print("📁 Check the 'assets/ventamin_dataset/stock_footage/' folder for generated backgrounds")

if __name__ == "__main__":
    main() 