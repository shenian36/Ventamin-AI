#!/usr/bin/env python3
"""
Test script to generate Sora-style videos based on product images
"""

import logging
from pathlib import Path
from src.generators.sora_style_generator import SoraStyleGenerator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main test function"""
    print("🎬 Testing Sora-Style Video Generation")
    print("=" * 50)
    
    try:
        # Initialize Sora-style generator
        generator = SoraStyleGenerator()
        print("✅ Sora-style generator initialized")
        
        # Generate videos for all Ventamin products
        print("\n🎥 Generating Sora-style videos from product images...")
        generated_videos = generator.generate_ventamin_videos()
        
        if generated_videos:
            print(f"\n🎉 Successfully generated {len(generated_videos)} videos:")
            for video_path in generated_videos:
                video_file = Path(video_path)
                if video_file.exists():
                    file_size = video_file.stat().st_size / 1024 / 1024
                    print(f"📁 {video_file.name} ({file_size:.2f} MB)")
                else:
                    print(f"❌ Video file not found: {video_path}")
        else:
            print("❌ No videos were generated")
            
        print(f"\n📋 Total videos generated: {len(generated_videos)}")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        logger.exception("Test failed")

if __name__ == "__main__":
    main() 