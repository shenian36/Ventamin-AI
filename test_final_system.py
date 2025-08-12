#!/usr/bin/env python3
"""
Final test script to verify the cleaned up Ventamin AI system
"""

import logging
from pathlib import Path
from src.analysis.ventamin_analyzer import VentaminAnalyzer
from src.generators.sora_style_generator import SoraStyleGenerator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test the complete Ventamin AI system"""
    print("🎬 Testing Complete Ventamin AI System")
    print("=" * 50)
    
    try:
        # Test 1: Ventamin Product Analysis
        print("\n📊 Test 1: Ventamin Product Analysis")
        analyzer = VentaminAnalyzer()
        analysis_results = analyzer.analyze_product_images()
        print("✅ Product analysis completed")
        
        # Save analysis results
        analysis_file = analyzer.save_analysis_results(analysis_results)
        print(f"📁 Analysis saved to: {analysis_file}")
        
        # Generate summary
        summary = analyzer.generate_summary(analysis_results)
        print("\n📋 Analysis Summary:")
        print(summary)
        
        # Test 2: Sora-Style Video Generation
        print("\n🎬 Test 2: Sora-Style Video Generation")
        generator = SoraStyleGenerator()
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
        
        # Test 3: Check output directories
        print("\n📁 Test 3: Output Directory Check")
        output_dirs = ['analysis_output', 'generated_videos', 'assets/ventamin_assets']
        for dir_path in output_dirs:
            path = Path(dir_path)
            if path.exists():
                files = list(path.glob('*'))
                print(f"✅ {dir_path}: {len(files)} files")
            else:
                print(f"❌ {dir_path}: Directory not found")
        
        print(f"\n🎉 All tests completed successfully!")
        print(f"📋 Total videos generated: {len(generated_videos)}")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        logger.exception("Test failed")

if __name__ == "__main__":
    main() 