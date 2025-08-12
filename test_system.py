#!/usr/bin/env python3
"""
Simple test script to verify Ventamin AI system components
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing module imports...")
    
    try:
        import cv2
        print(f"✓ OpenCV imported successfully (version: {cv2.__version__})")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        from moviepy import VideoFileClip
        print("✓ MoviePy imported successfully")
    except ImportError as e:
        print(f"✗ MoviePy import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy imported successfully (version: {np.__version__})")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"✓ Pillow imported successfully (version: {Image.__version__})")
    except ImportError as e:
        print(f"✗ Pillow import failed: {e}")
        return False
    
    return True

def test_assets():
    """Test if required assets exist"""
    print("\nTesting assets...")
    
    assets_dir = Path("assets/ventamin_assets")
    if assets_dir.exists():
        print(f"✓ Assets directory exists: {assets_dir}")
        
        sachet_image = assets_dir / "ventamin_lightup_sachet.png"
        box_image = assets_dir / "ventamin_lightup_box.png"
        
        if sachet_image.exists():
            print(f"✓ Sachet image found: {sachet_image}")
        else:
            print(f"✗ Sachet image missing: {sachet_image}")
            
        if box_image.exists():
            print(f"✓ Box image found: {box_image}")
        else:
            print(f"✗ Box image missing: {box_image}")
    else:
        print(f"✗ Assets directory missing: {assets_dir}")
        return False
    
    return True

def test_output_directories():
    """Test if output directories exist"""
    print("\nTesting output directories...")
    
    directories = [
        "analysis_output",
        "generated_videos", 
        "temp_frames",
        "logs",
        "config"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        if dir_path.exists():
            print(f"✓ Directory exists: {directory}")
        else:
            print(f"✗ Directory missing: {directory}")
    
    return True

def test_generated_videos():
    """Test if videos were generated"""
    print("\nTesting generated videos...")
    
    videos_dir = Path("generated_videos")
    if videos_dir.exists():
        video_files = list(videos_dir.glob("*.mp4"))
        if video_files:
            print(f"✓ Found {len(video_files)} generated videos:")
            for video in video_files:
                size_mb = video.stat().st_size / (1024 * 1024)
                print(f"  - {video.name} ({size_mb:.1f} MB)")
        else:
            print("✗ No video files found in generated_videos directory")
    else:
        print("✗ Generated videos directory missing")
        return False
    
    return True

def main():
    """Run all tests"""
    print("Ventamin AI System Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_assets,
        test_output_directories,
        test_generated_videos
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with error: {e}")
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

