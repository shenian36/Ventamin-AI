"""
Ventamin Product Analyzer
Specialized analysis for Ventamin products to generate detailed video generation prompts.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class VentaminAnalyzer:
    def __init__(self):
        self.assets_path = Path("assets/ventamin_assets")
        self.products = {
            "ventamin_lightup_sachet": {
                "name": "Ventamin Light Up Sachet",
                "description": "Golden sachet with black text reading 'Ventamin' and 'LIGHT UP'",
                "type": "skincare_supplement",
                "features": ["golden_packaging", "vertical_text", "premium_aesthetic"],
                "target_audience": "beauty_conscious_consumers",
                "benefits": ["skin_brightening", "glow_enhancement", "natural_ingredients"]
            },
            "ventamin_lightup_box": {
                "name": "Ventamin Light Up Box",
                "description": "Bright yellow box with 'LIGHT UP' title and botanical beverage description",
                "type": "skincare_supplement",
                "features": ["bright_yellow_packaging", "dual_language", "botanical_ingredients"],
                "target_audience": "health_conscious_consumers",
                "benefits": ["lemon_powder", "fern_extract", "botanical_beverage"]
            }
        }
    
    def analyze_product_images(self) -> Dict:
        """Analyze Ventamin product images and generate detailed prompts"""
        logger.info("Analyzing Ventamin product images...")
        
        analysis_results = {
            "product_analysis": {},
            "video_generation_prompts": {},
            "voice_over_scripts": {},
            "ai_recommendations": {
                "video_generation_guidelines": {
                    "style_preference": "luxury_beauty_cinematic",
                    "color_scheme": ["#FFD700", "#7FDBFF", "#FFFFFF", "#000000"],
                    "typography": "minimalist_serif",
                    "animation_style": "smooth_elegant",
                    "pacing": "slow_to_moderate",
                    "visual_effects": ["shallow_depth_of_field", "golden_hour_lighting", "particle_effects"]
                },
                "technical_specifications": {
                    "resolution": "4K",
                    "frame_rate": 24,
                    "duration": "8-15 seconds",
                    "format": "MP4",
                    "encoding": "H.264"
                },
                "content_optimization": {
                    "key_messages": [
                        "Illuminate Your Skin's Radiance",
                        "Glow from within",
                        "Doctor-formulated hydration",
                        "Try risk-free"
                    ],
                    "visual_elements": [
                        "Golden bioluminescent particles",
                        "Marble surface reflections",
                        "Manicured hands",
                        "Botanical elements"
                    ],
                    "brand_elements": [
                        "Ventamin logo",
                        "Light Up branding",
                        "90-Day Guarantee",
                        "@ventamin.global watermark"
                    ]
                }
            }
        }
        
        # Analyze each product
        for product_id, product_info in self.products.items():
            analysis_results["product_analysis"][product_id] = self._analyze_single_product(product_info)
            analysis_results["video_generation_prompts"][product_id] = self._generate_video_prompt(product_info)
            analysis_results["voice_over_scripts"][product_id] = self._generate_voice_over_script(product_info)
        
        return analysis_results
    
    def _analyze_single_product(self, product_info: Dict) -> Dict:
        """Analyze a single product and extract key features"""
        return {
            "product_name": product_info["name"],
            "product_type": product_info["type"],
            "packaging_features": product_info["features"],
            "target_audience": product_info["target_audience"],
            "key_benefits": product_info["benefits"],
            "visual_characteristics": {
                "primary_colors": ["golden", "yellow"] if "golden" in product_info["features"] else ["yellow"],
                "text_elements": ["Ventamin", "LIGHT UP"],
                "packaging_style": "premium" if "premium_aesthetic" in product_info["features"] else "modern",
                "brand_positioning": "luxury_skincare"
            }
        }
    
    def _generate_video_prompt(self, product_info: Dict) -> Dict:
        """Generate detailed video generation prompts"""
        if "golden" in product_info["features"]:
            # Sachet version - more luxury focused
            return {
                "main_prompt": """Ultra-realistic 4K beauty product advertisement. A sleek matte-white 'Glow Revival' light-up box (30cm x 15cm) rests center-frame on a reflective marble surface. Soft golden hour lighting illuminates subtle condensation droplets on the box. A manicured hand enters from screen right, elegantly shaking a cylindrical light-up stick that emits a warm, pulsating amber glow from its core. As the stick moves, radiant light particles cascade onto the box's surface, making the product logo instantly illuminate in synchronized gold light. Background features blurred botanical elements (eucalyptus leaves, rose petals) with shallow depth of field. Cinematic close-up shot, smooth motion, luxury aesthetic with champagne gold accents. End frame: Text overlay 'Illuminate Your Skin's Radiance' in minimalist serif font.""",
                
                "detailed_breakdown": {
                    "scene_setup": "Reflective marble surface with golden hour lighting",
                    "main_product": "Sleek matte-white 'Glow Revival' light-up box",
                    "interaction": "Manicured hand with cylindrical light-up stick",
                    "visual_effects": "Radiant light particles cascading onto box surface",
                    "background": "Blurred botanical elements (eucalyptus leaves, rose petals)",
                    "camera_style": "Cinematic close-up with shallow depth of field",
                    "color_palette": "Champagne gold accents with luxury aesthetic",
                    "text_overlay": "Illuminate Your Skin's Radiance in minimalist serif font"
                }
            }
        else:
            # Box version - more vibrant and modern
            return {
                "main_prompt": """Vibrant 4K beauty product advertisement. A bright yellow 'Ventamin Light Up' box (20cm x 12cm) sits prominently on a clean white surface. Dynamic lighting creates soft shadows and highlights the box's modern design. A professional hand enters frame, holding a glowing amber stick that pulses with warm light. As the stick approaches the box, golden bioluminescent particles flow from the stick to the box, causing the 'LIGHT UP' text to glow with synchronized brilliance. Background features soft-focus botanical elements (lemon slices, fern leaves) with artistic bokeh. Modern, clean aesthetic with high contrast. End frame: Bold 'Ventamin Light Up' logo with tagline 'Brighten Your Day' in clean sans-serif font.""",
                
                "detailed_breakdown": {
                    "scene_setup": "Clean white surface with dynamic lighting",
                    "main_product": "Bright yellow 'Ventamin Light Up' box",
                    "interaction": "Professional hand with glowing amber stick",
                    "visual_effects": "Golden bioluminescent particles flowing to box",
                    "background": "Soft-focus botanical elements (lemon slices, fern leaves)",
                    "camera_style": "Modern with artistic bokeh",
                    "color_palette": "Bright yellow with high contrast",
                    "text_overlay": "Ventamin Light Up logo with 'Brighten Your Day' tagline"
                }
            }
    
    def _generate_voice_over_script(self, product_info: Dict) -> Dict:
        """Generate detailed voice-over scripts with timing"""
        if "golden" in product_info["features"]:
            # Luxury sachet version
            return {
                "script_title": "Ventamin Light Up Skincare Ad - 8s",
                "total_duration": 8,
                "segments": [
                    {
                        "time": "0-2s",
                        "text": "Glow from within!",
                        "visual_cue": "Box illuminates brighter",
                        "tone": "aspirational"
                    },
                    {
                        "time": "2-5s", 
                        "text": "Ventamin Light Up: Brightens skin, fades dark spots",
                        "visual_cue": "Particles highlight cheekbone + dissolve dark spot on skin",
                        "tone": "informative"
                    },
                    {
                        "time": "5-7s",
                        "text": "Doctor-formulated hydration",
                        "visual_cue": "Skin texture plumps with dewy shine",
                        "tone": "trustworthy"
                    },
                    {
                        "time": "7-8s",
                        "text": "Try risk-free!",
                        "visual_cue": "Product box zooms + '90-Day Guarantee' text overlay",
                        "tone": "encouraging"
                    }
                ],
                "text_elements": {
                    "watermark": "@ventamin.global",
                    "end_screen": "Ventamin logo + #LightUpYourGlow",
                    "style": "minimalist serif font"
                }
            }
        else:
            # Vibrant box version
            return {
                "script_title": "Ventamin Light Up Beverage Ad - 10s",
                "total_duration": 10,
                "segments": [
                    {
                        "time": "0-2s",
                        "text": "Illuminate your day!",
                        "visual_cue": "Box begins to glow",
                        "tone": "energetic"
                    },
                    {
                        "time": "2-5s",
                        "text": "Ventamin Light Up: Botanical blend for radiant skin",
                        "visual_cue": "Particles flow from stick to box",
                        "tone": "educational"
                    },
                    {
                        "time": "5-8s",
                        "text": "Lemon powder with fern extract",
                        "visual_cue": "Ingredients highlighted with particle effects",
                        "tone": "natural"
                    },
                    {
                        "time": "8-10s",
                        "text": "30 sachets for your glow journey!",
                        "visual_cue": "Box opens to reveal sachets",
                        "tone": "inviting"
                    }
                ],
                "text_elements": {
                    "watermark": "@ventamin.global",
                    "end_screen": "Ventamin Light Up logo + #GlowJourney",
                    "style": "clean sans-serif font"
                }
            }
    
    def save_analysis_results(self, results: Dict) -> str:
        """Save analysis results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path("analysis_output") / f"ventamin_analysis_{timestamp}.json"
        
        # Ensure directory exists
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Ventamin analysis results saved to: {output_file}")
        return str(output_file)
    
    def generate_summary(self, results: Dict) -> str:
        """Generate a human-readable summary of the analysis"""
        summary = []
        summary.append("🎬 Ventamin Product Video Analysis Summary")
        summary.append("=" * 50)
        summary.append("")
        
        for product_id, product_analysis in results["product_analysis"].items():
            summary.append(f"📦 Product: {product_analysis['product_name']}")
            summary.append(f"🎯 Target: {product_analysis['target_audience']}")
            summary.append(f"✨ Key Benefits: {', '.join(product_analysis['key_benefits'])}")
            summary.append("")
        
        summary.append("🎥 Video Generation Guidelines:")
        guidelines = results["ai_recommendations"]["video_generation_guidelines"]
        summary.append(f"   • Style: {guidelines['style_preference']}")
        summary.append(f"   • Colors: {', '.join(guidelines['color_scheme'])}")
        summary.append(f"   • Duration: {results['ai_recommendations']['technical_specifications']['duration']}")
        summary.append("")
        
        summary.append("📝 Key Messages:")
        for message in results["ai_recommendations"]["content_optimization"]["key_messages"]:
            summary.append(f"   • {message}")
        summary.append("")
        
        summary.append("🎨 Visual Elements:")
        for element in results["ai_recommendations"]["content_optimization"]["visual_elements"]:
            summary.append(f"   • {element}")
        
        return "\n".join(summary) 