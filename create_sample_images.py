#!/usr/bin/env python3
"""
Create sample product images for testing Sora-style video generation
"""

import cv2
import numpy as np
from pathlib import Path

def create_ventamin_sachet_image():
    """Create a sample Ventamin Light Up sachet image matching the actual product"""
    # Create a golden sachet image matching the actual product
    width, height = 400, 600
    
    # Create golden background with metallic sheen
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Fill with golden color (matching the actual product)
    image[:, :] = [255, 215, 0]  # Golden color
    
    # Add metallic texture effect
    for y in range(0, height, 15):
        for x in range(0, width, 15):
            # Create metallic sheen effect
            intensity = 255 - (x + y) % 30
            cv2.rectangle(image, (x, y), (x+10, y+10), [intensity, intensity-20, 0], -1)
    
    # Add crimped seals at top and bottom (like the actual sachet)
    # Top seal
    for x in range(50, width-50, 5):
        for y in range(20, 40, 3):
            cv2.circle(image, (x, y), 1, (0, 0, 0), -1)
    
    # Bottom seal
    for x in range(50, width-50, 5):
        for y in range(height-40, height-20, 3):
            cv2.circle(image, (x, y), 1, (0, 0, 0), -1)
    
    # Add text vertically (matching the actual product)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 2
    
    # Add "ventamin" text (lowercase as in actual product)
    text = "ventamin"
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = height // 2 - 30
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
    
    # Add "LIGHT UP" text (uppercase as in actual product)
    text2 = "LIGHT UP"
    text_size2 = cv2.getTextSize(text2, font, font_scale, thickness)[0]
    text_x2 = (width - text_size2[0]) // 2
    text_y2 = height // 2 + 30
    cv2.putText(image, text2, (text_x2, text_y2), font, font_scale, (0, 0, 0), thickness)
    
    return image

def create_ventamin_box_image():
    """Create a sample Ventamin Light Up box image matching the actual product"""
    # Create a bright yellow box image matching the actual product
    width, height = 600, 400
    
    # Create bright yellow background (matching the actual product)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Fill with bright yellow color (matching the actual product)
    image[:, :] = [0, 255, 255]  # Bright yellow in BGR
    
    # Add box border
    cv2.rectangle(image, (50, 50), (width-50, height-50), (0, 0, 0), 5)
    
    # Add text matching the actual product
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.5
    thickness = 4
    
    # Add "LIGHT UP" title (matching the actual product)
    text = "LIGHT UP"
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = 120
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
    
    # Add description (matching the actual product)
    font_scale_small = 0.8
    thickness_small = 2
    
    desc1 = "Botanical Beverage"
    desc2 = "Mix Lemon Powder with Fern Extract"
    
    text_size1 = cv2.getTextSize(desc1, font, font_scale_small, thickness_small)[0]
    text_x1 = (width - text_size1[0]) // 2
    text_y1 = 200
    cv2.putText(image, desc1, (text_x1, text_y1), font, font_scale_small, (0, 0, 0), thickness_small)
    
    text_size2 = cv2.getTextSize(desc2, font, font_scale_small, thickness_small)[0]
    text_x2 = (width - text_size2[0]) // 2
    text_y2 = 230
    cv2.putText(image, desc2, (text_x2, text_y2), font, font_scale_small, (0, 0, 0), thickness_small)
    
    # Add Malay text (matching the actual product)
    malay1 = "Campuran Minuman"
    malay2 = "Botani Serbuk Lemon dengan Ekstrak Paku Paks"
    
    text_size_m1 = cv2.getTextSize(malay1, font, font_scale_small-0.1, thickness_small)[0]
    text_x_m1 = (width - text_size_m1[0]) // 2
    text_y_m1 = 260
    cv2.putText(image, malay1, (text_x_m1, text_y_m1), font, font_scale_small-0.1, (0, 0, 0), thickness_small)
    
    text_size_m2 = cv2.getTextSize(malay2, font, font_scale_small-0.2, thickness_small)[0]
    text_x_m2 = (width - text_size_m2[0]) // 2
    text_y_m2 = 285
    cv2.putText(image, malay2, (text_x_m2, text_y_m2), font, font_scale_small-0.2, (0, 0, 0), thickness_small)
    
    # Add quantity info (matching the actual product)
    quantity = "3g x 30 sachets"
    text_size_q = cv2.getTextSize(quantity, font, font_scale_small, thickness_small)[0]
    text_x_q = 100
    text_y_q = 330
    cv2.putText(image, quantity, (text_x_q, text_y_q), font, font_scale_small, (0, 0, 0), thickness_small)
    
    # Add Malay quantity
    quantity_malay = "3g x 30 paket"
    text_size_qm = cv2.getTextSize(quantity_malay, font, font_scale_small-0.1, thickness_small)[0]
    text_x_qm = 100
    text_y_qm = 350
    cv2.putText(image, quantity_malay, (text_x_qm, text_y_qm), font, font_scale_small-0.1, (0, 0, 0), thickness_small)
    
    # Add Ventamin logo (matching the actual product)
    # Create infinity loop logo
    logo_center_x = width - 100
    logo_center_y = 340
    
    # Draw infinity loop
    for i in range(0, 360, 10):
        angle = np.radians(i)
        x1 = logo_center_x - 20 + int(15 * np.cos(angle))
        y1 = logo_center_y - 10 + int(8 * np.sin(angle))
        x2 = logo_center_x + 20 + int(15 * np.cos(angle))
        y2 = logo_center_y + 10 + int(8 * np.sin(angle))
        cv2.circle(image, (x1, y1), 2, (255, 255, 255), -1)
        cv2.circle(image, (x2, y2), 2, (255, 255, 255), -1)
    
    # Add "ventamin" text below logo
    logo_text = "ventamin"
    font_scale_logo = 1.0
    text_size_logo = cv2.getTextSize(logo_text, font, font_scale_logo, thickness_small)[0]
    text_x_logo = logo_center_x - text_size_logo[0] // 2
    text_y_logo = 370
    cv2.putText(image, logo_text, (text_x_logo, text_y_logo), font, font_scale_logo, (255, 255, 255), thickness_small)
    
    return image

def main():
    """Create sample product images"""
    print("🎨 Creating sample Ventamin product images...")
    
    # Create assets directory
    assets_dir = Path("assets/ventamin_assets")
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sachet image
    print("📦 Creating Ventamin Light Up sachet image...")
    sachet_image = create_ventamin_sachet_image()
    sachet_path = assets_dir / "ventamin_lightup_sachet.png"
    cv2.imwrite(str(sachet_path), sachet_image)
    print(f"✅ Sachet image saved: {sachet_path}")
    
    # Create box image
    print("📦 Creating Ventamin Light Up box image...")
    box_image = create_ventamin_box_image()
    box_path = assets_dir / "ventamin_lightup_box.png"
    cv2.imwrite(str(box_path), box_image)
    print(f"✅ Box image saved: {box_path}")
    
    print("\n🎉 Sample product images created successfully!")
    print("📁 Images saved in: assets/ventamin_assets/")
    print("   - ventamin_lightup_sachet.png")
    print("   - ventamin_lightup_box.png")

if __name__ == "__main__":
    main() 