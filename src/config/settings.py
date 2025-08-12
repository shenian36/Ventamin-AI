"""
Configuration settings for Advanced AI Video Generator
"""

import os
from pathlib import Path

# Project paths - fixed for clean structure
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Go up to project root
SRC_DIR = PROJECT_ROOT / "src"
ASSETS_DIR = PROJECT_ROOT / "assets"
VENTAMIN_ASSETS_DIR = ASSETS_DIR / "ventamin_assets"
STOCK_FOOTAGE_DIR = ASSETS_DIR / "stock_footage"
GENERATED_ADS_DIR = ASSETS_DIR / "generated_ads"
OUTPUT_DIR = PROJECT_ROOT / "output"
TEMP_DIR = PROJECT_ROOT / "temp"

# Create directories if they don't exist
for directory in [SRC_DIR, ASSETS_DIR, VENTAMIN_ASSETS_DIR, STOCK_FOOTAGE_DIR, GENERATED_ADS_DIR, OUTPUT_DIR, TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Competitor information
COMPETITORS = {
    "G1": {
        "name": "G1",
        "facebook_page": "https://www.facebook.com/ads/library/?active_status=all&ad_type=all&country=US&view_all_page_id=G1",
        "keywords": ["supplements", "health", "wellness", "vitamins"]
    },
    "M8": {
        "name": "M8", 
        "facebook_page": "https://www.facebook.com/ads/library/?active_status=all&ad_type=all&country=US&view_all_page_id=M8",
        "keywords": ["supplements", "health", "wellness", "vitamins"]
    },
    "Loop": {
        "name": "Loop",
        "facebook_page": "https://www.facebook.com/ads/library/?active_status=all&ad_type=all&country=US&view_all_page_id=Loop",
        "keywords": ["supplements", "health", "wellness", "vitamins"]
    }
}

# Ventamin brand information
VENTAMIN_BRAND = {
    "name": "Ventamin",
    "primary_color": "#4A90E2",
    "secondary_color": "#F5A623",
    "font_family": "Arial, sans-serif",
    "tagline": "Your Health, Our Priority",
    "website": "https://ventamin.com"
}

# Ad generation settings
AD_SETTINGS = {
    "static_image": {
        "width": 1080,
        "height": 1080,
        "format": "PNG"
    },
    "ugc_video": {
        "width": 1080,
        "height": 1920,  # Vertical for mobile
        "duration": 15,  # seconds
        "fps": 30,
        "format": "MP4"
    },
    "face_ugc_video": {
        "width": 1080,
        "height": 1920,
        "duration": 15,
        "fps": 30,
        "format": "MP4"
    }
}

# AI model settings
AI_MODELS = {
    "image_generation": "stabilityai/stable-diffusion-2-1",
    "text_generation": "gpt2",
    "image_analysis": "microsoft/DialoGPT-medium"
}

# Facebook Ad Library settings
FACEBOOK_SETTINGS = {
    "base_url": "https://www.facebook.com/ads/library",
    "search_delay": 2,  # seconds between requests
    "max_ads_per_competitor": 50,
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Analysis settings
ANALYSIS_SETTINGS = {
    "min_engagement_rate": 0.01,  # 1%
    "top_performing_threshold": 0.05,  # 5%
    "pattern_similarity_threshold": 0.7
}

# Output settings
OUTPUT_SETTINGS = {
    "save_analysis": True,
    "save_generated_ads": True,
    "create_report": True,
    "output_format": "json"
} 