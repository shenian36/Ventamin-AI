"""
Dataset Loader for Ventamin Product Assets
Loads and manages Ventamin product dataset for AI video generation
"""

import json
import os
from pathlib import Path
from PIL import Image
import logging
from typing import Dict, List, Optional

class VentaminDatasetLoader:
    def __init__(self, dataset_path: str = "assets/ventamin_dataset"):
        self.dataset_path = Path(dataset_path)
        self.config_path = self.dataset_path / "dataset_config.json"
        self.setup_logging()
        self.config = self.load_config()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_config(self) -> Dict:
        """Load dataset configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            self.logger.info(f"Loaded dataset config from {self.config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Error loading dataset config: {e}")
            return {}
    
    def get_available_assets(self, category: str) -> List[str]:
        """Get list of available assets in a category"""
        category_path = self.dataset_path / category
        if not category_path.exists():
            return []
        
        assets = []
        for file_path in category_path.iterdir():
            if file_path.is_file():
                assets.append(str(file_path))
        
        return assets
    
    def get_product_images(self) -> List[str]:
        """Get all product images"""
        return self.get_available_assets("products")
    
    def get_lifestyle_images(self) -> List[str]:
        """Get all lifestyle images"""
        return self.get_available_assets("lifestyle")
    
    def get_testimonial_content(self) -> List[str]:
        """Get all testimonial content"""
        return self.get_available_assets("testimonials")
    
    def get_branding_assets(self) -> List[str]:
        """Get all branding assets"""
        return self.get_available_assets("branding")
    
    def get_stock_footage(self) -> List[str]:
        """Get all stock footage"""
        return self.get_available_assets("stock_footage")
    
    def get_videos(self) -> List[str]:
        """Get all video content"""
        return self.get_available_assets("videos")
    
    def get_assets_for_category(self, video_category: str) -> Dict[str, List[str]]:
        """Get required and optional assets for a video category"""
        if "content_categories" not in self.config:
            return {"required": [], "optional": []}
        
        category_config = self.config["content_categories"].get(video_category, {})
        required_assets = category_config.get("required_assets", [])
        optional_assets = category_config.get("optional_assets", [])
        
        assets = {
            "required": {},
            "optional": {}
        }
        
        # Load required assets
        for asset_type in required_assets:
            assets["required"][asset_type] = self.get_available_assets(asset_type)
        
        # Load optional assets
        for asset_type in optional_assets:
            assets["optional"][asset_type] = self.get_available_assets(asset_type)
        
        return assets
    
    def get_brand_guidelines(self) -> Dict:
        """Get brand guidelines from config"""
        return self.config.get("brand_guidelines", {})
    
    def get_ai_learning_parameters(self) -> Dict:
        """Get AI learning parameters from config"""
        return self.config.get("ai_learning_parameters", {})
    
    def validate_asset(self, file_path: str) -> bool:
        """Validate if an asset file is supported"""
        if not os.path.exists(file_path):
            return False
        
        file_ext = Path(file_path).suffix.lower()
        supported_formats = []
        
        # Get supported formats from config
        for category_info in self.config.get("folder_structure", {}).values():
            supported_formats.extend(category_info.get("supported_formats", []))
        
        return file_ext[1:] in supported_formats
    
    def get_asset_info(self, file_path: str) -> Dict:
        """Get information about an asset file"""
        if not self.validate_asset(file_path):
            return {}
        
        try:
            file_path_obj = Path(file_path)
            file_info = {
                "filename": file_path_obj.name,
                "extension": file_path_obj.suffix.lower(),
                "size": os.path.getsize(file_path),
                "category": self.get_asset_category(file_path)
            }
            
            # Get image dimensions if it's an image
            if file_info["extension"] in [".jpg", ".jpeg", ".png", ".webp"]:
                try:
                    with Image.open(file_path) as img:
                        file_info["dimensions"] = img.size
                        file_info["mode"] = img.mode
                except Exception as e:
                    self.logger.warning(f"Could not get image info for {file_path}: {e}")
            
            return file_info
            
        except Exception as e:
            self.logger.error(f"Error getting asset info for {file_path}: {e}")
            return {}
    
    def get_asset_category(self, file_path: str) -> str:
        """Determine which category an asset belongs to"""
        file_path_obj = Path(file_path)
        
        # Check if file is in any of the dataset categories
        for category in ["products", "lifestyle", "testimonials", "branding", "stock_footage", "videos"]:
            category_path = self.dataset_path / category
            if category_path in file_path_obj.parents:
                return category
        
        return "unknown"
    
    def get_dataset_summary(self) -> Dict:
        """Get a summary of the dataset"""
        summary = {
            "total_assets": 0,
            "categories": {},
            "brand_guidelines": self.get_brand_guidelines(),
            "ai_parameters": self.get_ai_learning_parameters()
        }
        
        # Count assets in each category
        for category in ["products", "lifestyle", "testimonials", "branding", "stock_footage", "videos"]:
            assets = self.get_available_assets(category)
            summary["categories"][category] = {
                "count": len(assets),
                "files": assets
            }
            summary["total_assets"] += len(assets)
        
        return summary
    
    def add_asset(self, file_path: str, category: str) -> bool:
        """Add a new asset to the dataset"""
        try:
            if not self.validate_asset(file_path):
                self.logger.error(f"Unsupported file format: {file_path}")
                return False
            
            category_path = self.dataset_path / category
            category_path.mkdir(exist_ok=True)
            
            # Copy file to appropriate category
            import shutil
            filename = Path(file_path).name
            destination = category_path / filename
            
            shutil.copy2(file_path, destination)
            self.logger.info(f"Added {filename} to {category} category")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding asset {file_path}: {e}")
            return False
    
    def remove_asset(self, file_path: str) -> bool:
        """Remove an asset from the dataset"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                self.logger.info(f"Removed asset: {file_path}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error removing asset {file_path}: {e}")
            return False

def main():
    """Test the dataset loader"""
    loader = VentaminDatasetLoader()
    
    print("📁 Ventamin Dataset Loader Test")
    print("=" * 40)
    
    # Get dataset summary
    summary = loader.get_dataset_summary()
    print(f"📊 Dataset Summary:")
    print(f"   Total assets: {summary['total_assets']}")
    for category, info in summary["categories"].items():
        print(f"   {category}: {info['count']} files")
    
    print(f"\n🎨 Brand Guidelines:")
    brand_guidelines = summary["brand_guidelines"]
    print(f"   Primary colors: {brand_guidelines.get('primary_colors', [])}")
    print(f"   Brand voice: {brand_guidelines.get('brand_voice', 'Not specified')}")
    
    print(f"\n🤖 AI Learning Parameters:")
    ai_params = summary["ai_parameters"]
    print(f"   Style preferences: {len(ai_params.get('style_preferences', []))} items")
    print(f"   Content themes: {len(ai_params.get('content_themes', []))} items")
    
    print("\n✅ Dataset loader test completed!")

if __name__ == "__main__":
    main() 