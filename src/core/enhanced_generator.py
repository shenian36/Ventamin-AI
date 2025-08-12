#!/usr/bin/env python3
"""
Enhanced Ventamin AI Video Generator with Audio
Professional-grade video generation with modern visual effects and sound
"""

import json
import time
import logging
from datetime import datetime
from pathlib import Path
import os
import random
import math
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import io
import wave
import struct

# Try to import enhanced libraries
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from moviepy import ImageSequenceClip, VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip, ImageClip, ColorClip
    from moviepy.audio.AudioClip import AudioClip
    from moviepy.audio.fx import all as audio_fx
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

from config import GENERATED_ADS_DIR, VENTAMIN_BRAND
from src.utils.dataset_loader import VentaminDatasetLoader

class EnhancedVentaminVideoGenerator:
    def __init__(self):
        self.setup_logging()
        self.dataset_loader = VentaminDatasetLoader()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Enhanced settings
        self.resolution = (1080, 1920)  # Vertical format for social media
        self.fps = 60  # Higher FPS for smoother animations
        self.sample_rate = 44100
        self.audio_duration = 15  # Default video duration in seconds
        
        # Professional color palette
        self.colors = {
            'primary': (74, 144, 226),      # Ventamin blue
            'secondary': (255, 255, 255),    # White
            'accent': (255, 193, 7),         # Gold accent
            'dark': (33, 37, 41),            # Dark text
            'light': (248, 249, 250),        # Light background
            'success': (40, 167, 69),        # Green for success
            'warning': (255, 193, 7),        # Yellow for highlights
            'gradient_start': (74, 144, 226),
            'gradient_end': (52, 152, 219)
        }
        
        # Professional fonts (fallback to system fonts)
        self.fonts = {
            'title': self.get_font(72, 'bold'),
            'headline': self.get_font(56, 'bold'),
            'subtitle': self.get_font(42, 'normal'),
            'body': self.get_font(32, 'normal'),
            'caption': self.get_font(24, 'normal')
        }
        
        # Animation settings
        self.animation_duration = 0.5  # seconds for smooth transitions
        self.easing_functions = {
            'ease_in': lambda x: x * x,
            'ease_out': lambda x: 1 - (1 - x) * (1 - x),
            'ease_in_out': lambda x: x * x * (3 - 2 * x),
            'bounce': lambda x: 1 - (math.cos(x * math.pi * 2) * math.exp(-x * 3))
        }
        
        # Audio settings
        self.audio_settings = {
            'background_music_volume': 0.3,
            'voiceover_volume': 0.8,
            'sound_effects_volume': 0.6,
            'fade_in_duration': 0.5,
            'fade_out_duration': 1.0
        }
        
        # Load dataset
        self.load_dataset_images()
        
        # Product information
        self.products = {
            "light_up": {
                "name": "Ventamin Light Up",
                "tagline": "Illuminate Your Natural Radiance",
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
                    "Boosts collagen production",
                    "Improves skin texture",
                    "Enhances natural glow",
                    "Reduces dark spots",
                    "Brightens complexion"
                ],
                "voiceover_script": [
                    {"text": "Tired of dull, lackluster skin?", "duration": 2.5},
                    {"text": "Introducing Ventamin Light Up", "duration": 2.0},
                    {"text": "The revolutionary formula that illuminates your natural radiance", "duration": 3.0},
                    {"text": "Fades pigmentation naturally", "duration": 2.0},
                    {"text": "Boosts collagen production", "duration": 2.0},
                    {"text": "And enhances your natural glow", "duration": 2.0},
                    {"text": "Transform your skin today", "duration": 1.5}
                ]
            },
            "clear_it": {
                "name": "Ventamin Clear It",
                "tagline": "Clear Skin, Clear Confidence",
                "headlines": [
                    "Clear Skin Solution",
                    "Banish Blemishes",
                    "Clear Confidence",
                    "Pure Skin Power"
                ],
                "emotional_hooks": [
                    "Frustrated by persistent breakouts?",
                    "Tired of hiding behind makeup?",
                    "Want clear skin that lasts?",
                    "Struggling with acne scars?"
                ],
                "benefits": [
                    "Reduces acne breakouts",
                    "Heals existing blemishes",
                    "Prevents future breakouts",
                    "Smooths skin texture",
                    "Reduces inflammation",
                    "Promotes healing"
                ],
                "voiceover_script": [
                    {"text": "Frustrated by persistent breakouts?", "duration": 2.5},
                    {"text": "Meet Ventamin Clear It", "duration": 2.0},
                    {"text": "The breakthrough formula that banishes blemishes", "duration": 3.0},
                    {"text": "Reduces acne breakouts", "duration": 2.0},
                    {"text": "Heals existing blemishes", "duration": 2.0},
                    {"text": "And prevents future breakouts", "duration": 2.0},
                    {"text": "Clear skin, clear confidence", "duration": 1.5}
                ]
            }
        }
        
        # Social proof and CTAs
        self.social_proof = "94% of users reported visible skin improvement in 12 weeks"
        self.cta_options = [
            "Get 90-Day Money-Back Guarantee",
            "Shop Now - Free Shipping",
            "Try It Today - Risk Free",
            "Start Your Transformation Now"
        ]
    
    def get_font(self, size, weight='normal'):
        """Get font with fallback options"""
        try:
            if weight == 'bold':
                return ImageFont.truetype("arialbd.ttf", size)
            else:
                return ImageFont.truetype("arial.ttf", size)
        except:
            try:
                return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
            except:
                return ImageFont.load_default()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
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
            # Load images with support for multiple formats
            for category in self.available_images.keys():
                images = self.dataset_loader.get_available_assets(category)
                self.available_images[category] = [
                    img for img in images if img.endswith(('.jpg', '.jpeg', '.png', '.avif'))
                ]
                self.logger.info(f"Loaded {len(self.available_images[category])} {category} images")
        except Exception as e:
            self.logger.warning(f"Error loading dataset images: {e}")
    
    def create_enhanced_gradient(self, size, colors, direction='vertical'):
        """Create professional gradient backgrounds"""
        width, height = size
        image = Image.new('RGB', size)
        draw = ImageDraw.Draw(image)
        
        if direction == 'vertical':
            for y in range(height):
                ratio = y / height
                r = int(colors[0][0] * (1 - ratio) + colors[1][0] * ratio)
                g = int(colors[0][1] * (1 - ratio) + colors[1][1] * ratio)
                b = int(colors[0][2] * (1 - ratio) + colors[1][2] * ratio)
                draw.line([(0, y), (width, y)], fill=(r, g, b))
        elif direction == 'radial':
            center_x, center_y = width // 2, height // 2
            max_distance = math.sqrt(center_x**2 + center_y**2)
            
            for y in range(height):
                for x in range(width):
                    distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                    ratio = distance / max_distance
                    r = int(colors[0][0] * (1 - ratio) + colors[1][0] * ratio)
                    g = int(colors[0][1] * (1 - ratio) + colors[1][1] * ratio)
                    b = int(colors[0][2] * (1 - ratio) + colors[1][2] * ratio)
                    image.putpixel((x, y), (r, g, b))
        
        return image
    
    def create_glass_morphism_effect(self, base_image, blur_radius=15, opacity=0.3):
        """Create modern glass morphism effect"""
        # Create blurred background
        blurred = base_image.filter(ImageFilter.GaussianBlur(blur_radius))
        
        # Create glass overlay
        glass = Image.new('RGBA', base_image.size, (255, 255, 255, int(255 * opacity)))
        
        # Combine with original
        result = Image.alpha_composite(base_image.convert('RGBA'), glass)
        return result
    
    def create_animated_text(self, text, font, color, position, frame_num, total_frames, animation_type='fade_in'):
        """Create animated text with various effects"""
        # Create text image
        text_img = Image.new('RGBA', self.resolution, (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_img)
        
        # Calculate animation progress
        progress = frame_num / total_frames
        
        if animation_type == 'fade_in':
            opacity = int(255 * self.easing_functions['ease_in'](progress))
        elif animation_type == 'slide_up':
            opacity = int(255 * self.easing_functions['ease_out'](progress))
            y_offset = int(50 * (1 - self.easing_functions['ease_out'](progress)))
            position = (position[0], position[1] - y_offset)
        elif animation_type == 'typewriter':
            opacity = 255
            char_count = int(len(text) * self.easing_functions['ease_in'](progress))
            text = text[:char_count]
        elif animation_type == 'bounce':
            opacity = 255
            scale = 1 + 0.2 * self.easing_functions['bounce'](progress)
            # Scale effect would be applied to font size
        
        # Draw text with shadow
        x, y = position
        shadow_offset = 3
        
        # Draw shadow
        draw.text((x + shadow_offset, y + shadow_offset), text, 
                 font=font, fill=(0, 0, 0, opacity // 2))
        
        # Draw main text
        draw.text((x, y), text, font=font, fill=(*color, opacity))
        
        return text_img
    
    def create_product_showcase_frame(self, product_info, frame_num, total_frames):
        """Create professional product showcase frame"""
        width, height = self.resolution
        
        # Create gradient background
        background = self.create_enhanced_gradient(
            self.resolution, 
            [self.colors['gradient_start'], self.colors['gradient_end']], 
            'vertical'
        )
        
        # Add glass morphism overlay
        background = self.create_glass_morphism_effect(background, blur_radius=10, opacity=0.1)
        
        # Create product circle
        product_circle = Image.new('RGBA', (400, 400), (0, 0, 0, 0))
        draw = ImageDraw.Draw(product_circle)
        
        # Draw glowing circle
        center = (200, 200)
        radius = 150
        
        # Outer glow
        for r in range(radius + 20, radius - 5, -5):
            alpha = int(50 * (1 - (r - radius) / 20))
            draw.ellipse([center[0] - r, center[1] - r, center[0] + r, center[1] + r], 
                        fill=(255, 255, 255, alpha))
        
        # Main circle
        draw.ellipse([center[0] - radius, center[1] - radius, 
                     center[0] + radius, center[1] + radius], 
                    fill=(255, 255, 255, 200), outline=(255, 255, 255, 255), width=3)
        
        # Add product text
        product_text = product_info['name']
        font = self.fonts['headline']
        bbox = draw.textbbox((0, 0), product_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = center[0] - text_width // 2
        text_y = center[1] + radius + 30
        
        draw.text((text_x, text_y), product_text, font=font, fill=self.colors['primary'])
        
        # Position product circle
        circle_x = (width - 400) // 2
        circle_y = height // 2 - 250
        
        # Combine with background
        result = background.copy()
        result.paste(product_circle, (circle_x, circle_y), product_circle)
        
        # Add animated tagline
        tagline = product_info['tagline']
        tagline_img = self.create_animated_text(
            tagline, self.fonts['subtitle'], self.colors['secondary'],
            (width // 2, height // 2 + 100), frame_num, total_frames, 'slide_up'
        )
        
        result = Image.alpha_composite(result.convert('RGBA'), tagline_img)
        
        return result
    
    def create_benefits_frame(self, benefits, frame_num, total_frames):
        """Create professional benefits frame with icons"""
        width, height = self.resolution
        
        # Create background
        background = Image.new('RGB', self.resolution, self.colors['light'])
        
        # Add subtle pattern
        draw = ImageDraw.Draw(background)
        for i in range(0, width, 50):
            for j in range(0, height, 50):
                if (i + j) % 100 == 0:
                    draw.rectangle([i, j, i + 2, j + 2], fill=self.colors['primary'])
        
        # Create benefits list
        benefits_y = height // 3
        for i, benefit in enumerate(benefits[:4]):
            # Animate each benefit
            benefit_progress = max(0, (frame_num - i * total_frames // 8) / (total_frames // 8))
            if benefit_progress > 0:
                # Create benefit item
                item_y = benefits_y + i * 120
                
                # Checkmark icon
                check_color = self.colors['success']
                check_opacity = int(255 * self.easing_functions['ease_in'](benefit_progress))
                
                # Draw checkmark
                check_size = 30
                check_x = 100
                check_y = item_y - 15
                
                draw.ellipse([check_x, check_y, check_x + check_size, check_y + check_size], 
                           fill=(*check_color, check_opacity), outline=(*check_color, check_opacity))
                
                # Draw checkmark symbol
                draw.text((check_x + 8, check_y + 2), "✓", font=self.fonts['body'], 
                         fill=(255, 255, 255, check_opacity))
                
                # Benefit text
                text_opacity = int(255 * self.easing_functions['ease_in'](benefit_progress))
                draw.text((check_x + 50, item_y), benefit, font=self.fonts['body'], 
                         fill=(*self.colors['dark'], text_opacity))
        
        # Add title
        title = "Key Benefits"
        title_img = self.create_animated_text(
            title, self.fonts['headline'], self.colors['primary'],
            (width // 2, 100), frame_num, total_frames, 'fade_in'
        )
        
        result = Image.alpha_composite(background.convert('RGBA'), title_img)
        return result
    
    def create_social_proof_frame(self, frame_num, total_frames):
        """Create compelling social proof frame"""
        width, height = self.resolution
        
        # Create background with gradient
        background = self.create_enhanced_gradient(
            self.resolution, 
            [self.colors['primary'], self.colors['gradient_end']], 
            'radial'
        )
        
        # Add floating elements
        draw = ImageDraw.Draw(background)
        for i in range(5):
            x = random.randint(50, width - 50)
            y = random.randint(50, height - 50)
            size = random.randint(20, 60)
            alpha = random.randint(30, 80)
            draw.ellipse([x - size, y - size, x + size, y + size], 
                        fill=(255, 255, 255, alpha))
        
        # Create testimonial card
        card_width, card_height = 600, 300
        card_x = (width - card_width) // 2
        card_y = (height - card_height) // 2
        
        # Card background with glass effect
        card = Image.new('RGBA', (card_width, card_height), (255, 255, 255, 200))
        card = self.create_glass_morphism_effect(card, blur_radius=5, opacity=0.8)
        
        # Add card to background
        background.paste(card, (card_x, card_y), card)
        
        # Add social proof text
        proof_text = self.social_proof
        proof_img = self.create_animated_text(
            proof_text, self.fonts['subtitle'], self.colors['secondary'],
            (width // 2, height // 2), frame_num, total_frames, 'typewriter'
        )
        
        result = Image.alpha_composite(background.convert('RGBA'), proof_img)
        return result
    
    def create_cta_frame(self, cta_text, frame_num, total_frames):
        """Create compelling call-to-action frame"""
        width, height = self.resolution
        
        # Create dynamic background
        background = self.create_enhanced_gradient(
            self.resolution, 
            [self.colors['primary'], self.colors['accent']], 
            'vertical'
        )
        
        # Add animated elements
        draw = ImageDraw.Draw(background)
        time_factor = frame_num / total_frames
        
        # Animated circles
        for i in range(3):
            angle = time_factor * 2 * math.pi + i * 2 * math.pi / 3
            x = width // 2 + int(200 * math.cos(angle))
            y = height // 2 + int(200 * math.sin(angle))
            size = 30 + int(20 * math.sin(time_factor * 4 * math.pi + i))
            alpha = 100 + int(50 * math.sin(time_factor * 2 * math.pi + i))
            draw.ellipse([x - size, y - size, x + size, y + size], 
                        fill=(255, 255, 255, alpha))
        
        # Create CTA button
        button_width, button_height = 400, 80
        button_x = (width - button_width) // 2
        button_y = height // 2 + 50
        
        # Button background with glow
        for r in range(button_height + 20, button_height - 5, -5):
            alpha = int(30 * (1 - (r - button_height) / 20))
            draw.rounded_rectangle([button_x - r//2, button_y - r//2, 
                                  button_x + button_width + r//2, button_y + button_height + r//2], 
                                 radius=20, fill=(255, 255, 255, alpha))
        
        # Main button
        draw.rounded_rectangle([button_x, button_y, button_x + button_width, button_y + button_height], 
                             radius=20, fill=self.colors['secondary'], outline=self.colors['accent'], width=3)
        
        # CTA text
        cta_img = self.create_animated_text(
            cta_text, self.fonts['headline'], self.colors['primary'],
            (width // 2, height // 2 + 90), frame_num, total_frames, 'bounce'
        )
        
        result = Image.alpha_composite(background.convert('RGBA'), cta_img)
        return result
    
    def generate_enhanced_video(self, product_type="light_up"):
        """Generate enhanced video with professional quality"""
        self.logger.info(f"🎬 Generating enhanced {product_type} video...")
        
        product_info = self.products[product_type]
        total_duration = 30  # 30 seconds for better engagement
        total_frames = total_duration * self.fps
        
        frames = []
        
        # Scene 1: Product Showcase (0-6 seconds)
        showcase_frames = int(6 * self.fps)
        for i in range(showcase_frames):
            frame = self.create_product_showcase_frame(product_info, i, showcase_frames)
            frames.append(frame)
        
        # Scene 2: Benefits (6-12 seconds)
        benefits_frames = int(6 * self.fps)
        for i in range(benefits_frames):
            frame = self.create_benefits_frame(product_info['benefits'], i, benefits_frames)
            frames.append(frame)
        
        # Scene 3: Social Proof (12-18 seconds)
        proof_frames = int(6 * self.fps)
        for i in range(proof_frames):
            frame = self.create_social_proof_frame(i, proof_frames)
            frames.append(frame)
        
        # Scene 4: CTA (18-24 seconds)
        cta_frames = int(6 * self.fps)
        for i in range(cta_frames):
            frame = self.create_cta_frame(self.cta_options[0], i, cta_frames)
            frames.append(frame)
        
        # Scene 5: Final Product (24-30 seconds)
        final_frames = int(6 * self.fps)
        for i in range(final_frames):
            frame = self.create_product_showcase_frame(product_info, i, final_frames)
            frames.append(frame)
        
        # Create video
        filename = f"enhanced_ventamin_{product_type}_{self.timestamp}.mp4"
        return self.create_video_from_frames(frames, filename, self.fps, product_type)
    
    def create_video_from_frames(self, frames, filename, fps=60, product_type="light_up"):
        """Create video from frames with enhanced quality and audio"""
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
                frame.save(frame_path, quality=95, optimize=True)
                frame_paths.append(str(frame_path))
            
            # Create video using MoviePy
            from moviepy import ImageSequenceClip
            
            clip = ImageSequenceClip(frame_paths, fps=fps)
            
            # Generate audio for the video
            audio_clip = self.generate_audio_for_video(product_type, clip.duration)
            
            # Combine video and audio
            if audio_clip is not None:
                clip = clip.set_audio(audio_clip)
            
            video_path = GENERATED_ADS_DIR / filename
            clip.write_videofile(str(video_path), fps=fps, codec='libx264', 
                               preset='slow', crf=18, audio_codec='aac')  # High quality settings with audio
            
            # Clean up temporary files
            for frame_path in frame_paths:
                os.remove(frame_path)
            temp_dir.rmdir()
            
            return str(video_path)
            
        except Exception as e:
            self.logger.error(f"Error creating video: {e}")
            return self._create_frame_sequence(frames, filename)
    
    def generate_audio_for_video(self, product_type, video_duration):
        """Generate audio track for the video including voiceover and background music"""
        try:
            # Create background music
            background_music = self.create_background_music(video_duration)
            
            # Create voiceover audio
            voiceover_audio = self.create_voiceover_audio(product_type, video_duration)
            
            # Combine audio tracks
            if background_music is not None and voiceover_audio is not None:
                # Mix background music and voiceover
                mixed_audio = self.mix_audio_tracks(background_music, voiceover_audio)
                return mixed_audio
            elif voiceover_audio is not None:
                return voiceover_audio
            elif background_music is not None:
                return background_music
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error generating audio: {e}")
            return None
    
    def create_background_music(self, duration):
        """Create background music using synthetic audio generation"""
        try:
            # Generate a simple background music track
            sample_rate = self.sample_rate
            num_samples = int(duration * sample_rate)
            
            # Create a simple melody using sine waves
            t = np.linspace(0, duration, num_samples)
            
            # Base frequency for a pleasant chord progression
            frequencies = [440, 554, 659, 784]  # A, C#, E, G
            amplitudes = [0.1, 0.08, 0.06, 0.04]  # Decreasing amplitudes for layering
            
            # Create layered harmonic background
            background = np.zeros(num_samples)
            for freq, amp in zip(frequencies, amplitudes):
                wave = amp * np.sin(2 * np.pi * freq * t)
                # Add some variation over time
                envelope = np.exp(-t / duration) * (1 - np.exp(-t / 0.5))
                wave *= envelope
                background += wave
            
            # Normalize and add fade effects
            background = background / np.max(np.abs(background)) * 0.3
            
            # Add fade in/out
            fade_samples = int(self.audio_settings['fade_in_duration'] * sample_rate)
            background[:fade_samples] *= np.linspace(0, 1, fade_samples)
            
            fade_out_samples = int(self.audio_settings['fade_out_duration'] * sample_rate)
            background[-fade_out_samples:] *= np.linspace(1, 0, fade_out_samples)
            
            # Convert to MoviePy AudioClip
            from moviepy.audio.AudioClip import AudioArrayClip
            audio_clip = AudioArrayClip(background.reshape(-1, 1), fps=sample_rate)
            
            return audio_clip
            
        except Exception as e:
            self.logger.error(f"Error creating background music: {e}")
            return None
    
    def create_voiceover_audio(self, product_type, video_duration):
        """Create voiceover audio using TTS or synthetic speech"""
        try:
            product_info = self.products.get(product_type, self.products["light_up"])
            voiceover_script = product_info.get("voiceover_script", [])
            
            if not voiceover_script:
                return None
            
            # Create a simple continuous voiceover
            total_duration = sum(segment["duration"] for segment in voiceover_script)
            
            # Generate a single continuous audio for the entire script
            combined_text = " ".join([segment["text"] for segment in voiceover_script])
            voiceover_audio = self.generate_synthetic_speech(combined_text, total_duration)
            
            if voiceover_audio is not None:
                # Return voiceover without volume adjustment for now
                return voiceover_audio
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error creating voiceover audio: {e}")
            return None
    
    def generate_synthetic_speech(self, text, duration):
        """Generate synthetic speech for given text and duration"""
        try:
            # Create a simple synthetic speech using frequency modulation
            sample_rate = self.sample_rate
            num_samples = int(duration * sample_rate)
            
            # Generate a speech-like waveform
            t = np.linspace(0, duration, num_samples)
            
            # Base frequency for speech (around 150-200 Hz)
            base_freq = 180
            
            # Add some variation to make it sound more natural
            freq_variation = 20 * np.sin(2 * np.pi * 2 * t)  # Slow frequency variation
            frequency = base_freq + freq_variation
            
            # Generate the speech waveform
            speech = np.sin(2 * np.pi * frequency * t)
            
            # Add some harmonics to make it sound more realistic
            harmonics = [
                (2, 0.3),  # Second harmonic
                (3, 0.2),  # Third harmonic
                (4, 0.1),  # Fourth harmonic
            ]
            
            for harmonic_mult, amplitude in harmonics:
                harmonic_freq = base_freq * harmonic_mult
                harmonic_wave = amplitude * np.sin(2 * np.pi * harmonic_freq * t)
                speech += harmonic_wave
            
            # Apply envelope to make it sound more natural
            envelope = np.exp(-t / duration) * (1 - np.exp(-t / 0.1))
            speech *= envelope
            
            # Normalize
            speech = speech / np.max(np.abs(speech)) * 0.5
            
            # Convert to MoviePy AudioClip
            from moviepy.audio.AudioClip import AudioArrayClip
            audio_clip = AudioArrayClip(speech.reshape(-1, 1), fps=sample_rate)
            
            return audio_clip
            
        except Exception as e:
            self.logger.error(f"Error generating synthetic speech: {e}")
            return None
    
    def mix_audio_tracks(self, background_music, voiceover_audio):
        """Mix background music and voiceover audio"""
        try:
            from moviepy.audio.AudioClip import CompositeAudioClip
            
            # Ensure both clips have the same duration
            max_duration = max(background_music.duration, voiceover_audio.duration)
            
            # Extend background music if needed
            if background_music.duration < max_duration:
                # Simple loop by repeating the audio
                loops_needed = int(max_duration / background_music.duration) + 1
                repeated_clips = [background_music] * loops_needed
                background_music = CompositeAudioClip(repeated_clips)
                # Trim to exact duration
                if background_music.duration > max_duration:
                    background_music = background_music.set_duration(max_duration)
            
            # Extend voiceover if needed
            if voiceover_audio.duration < max_duration:
                # Pad with silence
                from moviepy.audio.AudioClip import AudioClip
                silence = AudioClip(lambda t: 0, duration=max_duration - voiceover_audio.duration)
                voiceover_audio = CompositeAudioClip([voiceover_audio, silence.set_start(voiceover_audio.duration)])
            
            # Mix the audio tracks without volume adjustment for now
            mixed_audio = CompositeAudioClip([
                background_music,
                voiceover_audio
            ])
            
            return mixed_audio
            
        except Exception as e:
            self.logger.error(f"Error mixing audio tracks: {e}")
            return voiceover_audio  # Return voiceover if mixing fails
    
    def _create_frame_sequence(self, frames, filename):
        """Create a frame sequence when video creation fails"""
        try:
            sequence_dir = GENERATED_ADS_DIR / filename.replace('.mp4', '_frames')
            sequence_dir.mkdir(exist_ok=True)
            
            for i, frame in enumerate(frames):
                frame_path = sequence_dir / f"frame_{i:04d}.png"
                frame.save(frame_path, quality=95, optimize=True)
            
            metadata = {
                "filename": filename,
                "frame_count": len(frames),
                "fps": self.fps,
                "resolution": f"{self.resolution[0]}x{self.resolution[1]}",
                "format": "PNG sequence",
                "generated_at": datetime.now().isoformat(),
                "note": "Enhanced video generation"
            }
            
            metadata_file = sequence_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Enhanced frame sequence saved to {sequence_dir}")
            return str(sequence_dir)
            
        except Exception as e:
            self.logger.error(f"Error creating frame sequence: {e}")
            return None

def main():
    """Test the enhanced video generator"""
    generator = EnhancedVentaminVideoGenerator()
    
    print("🎬 Enhanced Ventamin AI Video Generator")
    print("=" * 50)
    print("Professional-grade video generation with:")
    print("• Smooth animations and transitions")
    print("• Glass morphism effects")
    print("• Dynamic gradients and backgrounds")
    print("• High-quality graphics and typography")
    print("• 60 FPS for ultra-smooth playback")
    print()
    
    # Generate enhanced video
    result = generator.generate_enhanced_video("light_up")
    
    if result:
        print(f"✅ Enhanced video generated: {result}")
        if os.path.exists(result):
            size_mb = os.path.getsize(result) / (1024 * 1024)
            print(f"📁 File size: {size_mb:.1f} MB")
    else:
        print("❌ Enhanced video generation failed")
    
    print("\n🎬 Enhanced video generation completed!")

if __name__ == "__main__":
    main() 