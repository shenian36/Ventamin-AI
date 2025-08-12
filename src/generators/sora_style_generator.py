"""
Sora-Style Video Generator
Generates videos based on product images, similar to OpenAI's Sora approach.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import random

# Import MoviePy components
try:
    from moviepy import VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip, ColorClip, ImageClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logging.warning("MoviePy not available for Sora-style generation")

logger = logging.getLogger(__name__)

class SoraStyleGenerator:
    def __init__(self):
        self.assets_path = Path("assets/ventamin_assets")
        self.output_dir = Path("generated_videos")
        self.output_dir.mkdir(exist_ok=True)
        
        # Video generation parameters
        self.resolution = (1920, 1080)  # Full HD
        self.fps = 30
        self.duration = 8  # 8 seconds
        
    def load_product_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load and process a product image"""
        try:
            full_path = self.assets_path / image_path
            if full_path.exists():
                image = cv2.imread(str(full_path))
                if image is not None:
                    # Convert BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    logger.info(f"✅ Loaded image: {image_path}")
                    return image
                else:
                    logger.error(f"❌ Failed to load image: {image_path}")
                    return None
            else:
                logger.warning(f"⚠️ Image not found: {full_path}")
                return None
        except Exception as e:
            logger.error(f"❌ Error loading image {image_path}: {e}")
            return None
    
    def create_product_showcase_video(self, image: np.ndarray, product_info: Dict) -> str:
        """Create a product showcase video based on the image"""
        if not MOVIEPY_AVAILABLE:
            logger.error("❌ MoviePy not available for video generation")
            return ""
        
        try:
            # Create video clips based on the product image
            clips = []
            
            # 1. Intro clip - Product reveal
            intro_clip = self._create_intro_clip(image, product_info)
            clips.append(intro_clip)
            
            # 2. Feature highlight clips
            feature_clips = self._create_feature_clips(image, product_info)
            clips.extend(feature_clips)
            
            # 3. Call-to-action clip
            cta_clip = self._create_cta_clip(product_info)
            clips.append(cta_clip)
            
            # Combine all clips
            final_video = self._combine_clips(clips)
            
            # Export video
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"sora_style_{product_info.get('name', 'product').replace(' ', '_').lower()}_{timestamp}.mp4"
            output_path = self.output_dir / output_filename
            
            final_video.write_videofile(
                str(output_path),
                fps=self.fps,
                codec='libx264',
                audio_codec='aac'
            )
            
            final_video.close()
            
            logger.info(f"✅ Sora-style video generated: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ Error creating Sora-style video: {e}")
            raise
    
    def _create_intro_clip(self, image: np.ndarray, product_info: Dict) -> VideoFileClip:
        """Create an intro clip with the product image"""
        # Resize image to fit video dimensions
        height, width = image.shape[:2]
        target_width, target_height = self.resolution
        
        # Calculate scaling to maintain aspect ratio
        scale = min(target_width / width, target_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized_image = cv2.resize(image, (new_width, new_height))
        
        # Create background
        background = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Center the product image
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        background[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image
        
        # Convert to MoviePy clip
        intro_clip = ImageClip(background).with_duration(2)
        
        # Add text overlay
        product_name = product_info.get('name', 'Ventamin Light Up')
        text_clip = TextClip(
            text=product_name,
            font_size=60,
            color='white',
            size=(target_width-100, None)
        ).with_position('center').with_duration(2)
        
        # Composite intro
        intro = CompositeVideoClip([intro_clip, text_clip])
        return intro
    
    def _create_feature_clips(self, image: np.ndarray, product_info: Dict) -> List[VideoFileClip]:
        """Create feature highlight clips"""
        clips = []
        
        # Get product features
        features = product_info.get('features', [])
        benefits = product_info.get('benefits', [])
        
        # Create feature clips
        for i, feature in enumerate(features[:3]):  # Limit to 3 features
            # Create background with product image
            height, width = image.shape[:2]
            target_width, target_height = self.resolution
            
            # Create a stylized background
            background = self._create_stylized_background(image, feature)
            
            # Create feature clip
            feature_clip = ImageClip(background).with_duration(2)
            
            # Add feature text
            feature_text = f"✨ {feature.replace('_', ' ').title()}"
            text_clip = TextClip(
                text=feature_text,
                font_size=50,
                color='white',
                size=(target_width-100, None)
            ).with_position('center').with_duration(2)
            
            # Composite feature
            feature_video = CompositeVideoClip([feature_clip, text_clip])
            clips.append(feature_video)
        
        return clips
    
    def _create_stylized_background(self, image: np.ndarray, feature: str) -> np.ndarray:
        """Create a stylized background based on the feature"""
        target_width, target_height = self.resolution
        
        # Create gradient background
        background = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Add gradient based on feature
        if 'golden' in feature or 'premium' in feature:
            # Golden gradient
            for y in range(target_height):
                ratio = y / target_height
                color = [
                    int(255 * (1 - ratio)),  # Red
                    int(215 * (1 - ratio)),  # Green
                    int(0)                    # Blue
                ]
                background[y, :] = color
        elif 'bright' in feature or 'yellow' in feature:
            # Bright yellow gradient
            for y in range(target_height):
                ratio = y / target_height
                color = [
                    int(255),                # Red
                    int(255 * (1 - ratio)),  # Green
                    int(0)                    # Blue
                ]
                background[y, :] = color
        else:
            # Default blue gradient
            for y in range(target_height):
                ratio = y / target_height
                color = [
                    int(0),                  # Red
                    int(100 * (1 - ratio)),  # Green
                    int(255 * (1 - ratio))   # Blue
                ]
                background[y, :] = color
        
        # Add product image overlay with transparency
        height, width = image.shape[:2]
        scale = min(target_width / width, target_height / height) * 0.6
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        resized_image = cv2.resize(image, (new_width, new_height))
        
        # Center the product image
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        
        # Blend image with background
        for y in range(new_height):
            for x in range(new_width):
                if y_offset + y < target_height and x_offset + x < target_width:
                    # Simple alpha blending
                    alpha = 0.7
                    background[y_offset + y, x_offset + x] = (
                        background[y_offset + y, x_offset + x] * (1 - alpha) +
                        resized_image[y, x] * alpha
                    ).astype(np.uint8)
        
        return background
    
    def _create_cta_clip(self, product_info: Dict) -> VideoFileClip:
        """Create call-to-action clip"""
        target_width, target_height = self.resolution
        
        # Create animated background
        background = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Create animated gradient
        for y in range(target_height):
            for x in range(target_width):
                # Create animated wave effect
                wave = np.sin(x * 0.01 + y * 0.005) * 0.5 + 0.5
                color = [
                    int(255 * wave),  # Red
                    int(100 * wave),  # Green
                    int(255)          # Blue
                ]
                background[y, x] = color
        
        # Create background clip
        cta_clip = ImageClip(background).with_duration(2)
        
        # Add CTA text
        cta_text = "Illuminate Your Skin's Radiance"
        text_clip = TextClip(
            text=cta_text,
            font_size=60,
            color='white',
            size=(target_width-100, None)
        ).with_position('center').with_duration(2)
        
        # Composite CTA
        cta = CompositeVideoClip([cta_clip, text_clip])
        return cta
    
    def _combine_clips(self, clips: List[VideoFileClip]) -> VideoFileClip:
        """Combine multiple clips into one video"""
        if not clips:
            raise ValueError("No clips to combine")
        
        # Concatenate clips
        final_video = clips[0]
        for clip in clips[1:]:
            final_video = final_video.with_duration(final_video.duration + clip.duration)
            final_video = CompositeVideoClip([
                final_video, 
                clip.with_start(final_video.duration - clip.duration)
            ])
        
        return final_video
    
    def generate_video_from_image(self, image_path: str, product_info: Dict) -> str:
        """Generate a Sora-style video from a product image"""
        logger.info(f"🎬 Generating Sora-style video for {image_path}")
        
        # Load the product image
        image = self.load_product_image(image_path)
        if image is None:
            logger.error(f"❌ Could not load image: {image_path}")
            return ""
        
        # Create the video
        video_path = self.create_product_showcase_video(image, product_info)
        
        return video_path
    
    def generate_ventamin_videos(self) -> List[str]:
        """Generate videos for all Ventamin products"""
        ventamin_products = {
            "ventamin_lightup_sachet.png": {
                "name": "Ventamin Light Up Sachet",
                "features": ["golden_packaging", "vertical_text", "premium_aesthetic"],
                "benefits": ["skin_brightening", "glow_enhancement", "natural_ingredients"],
                "target_audience": "beauty_conscious_consumers"
            },
            "ventamin_lightup_box.png": {
                "name": "Ventamin Light Up Box", 
                "features": ["bright_yellow_packaging", "dual_language", "botanical_ingredients"],
                "benefits": ["lemon_powder", "fern_extract", "botanical_beverage"],
                "target_audience": "health_conscious_consumers"
            }
        }
        
        generated_videos = []
        
        for image_path, product_info in ventamin_products.items():
            try:
                video_path = self.generate_video_from_image(image_path, product_info)
                if video_path:
                    generated_videos.append(video_path)
                    logger.info(f"✅ Generated video: {video_path}")
            except Exception as e:
                logger.error(f"❌ Error generating video for {image_path}: {e}")
        
        return generated_videos 