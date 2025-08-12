"""
Facebook Ad Library Scraper
Extracts advertising data from Facebook Ad Library for competitor analysis
"""

import time
import json
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import logging

from config import FACEBOOK_SETTINGS, COMPETITORS

class FacebookAdScraper:
    def __init__(self):
        self.driver = None
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('facebook_scraper.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_driver(self):
        """Initialize Chrome WebDriver"""
        try:
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')  # Run in background
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument(f'--user-agent={FACEBOOK_SETTINGS["user_agent"]}')
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
            self.logger.info("Chrome WebDriver initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize WebDriver: {e}")
            return False
    
    def scrape_competitor_ads(self, competitor_name):
        """Scrape ads for a specific competitor"""
        competitor_data = COMPETITORS.get(competitor_name)
        if not competitor_data:
            self.logger.error(f"Competitor {competitor_name} not found in configuration")
            return []
        
        ads_data = []
        try:
            # Navigate to Facebook Ad Library
            self.driver.get(competitor_data["facebook_page"])
            time.sleep(FACEBOOK_SETTINGS["search_delay"])
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Scroll to load more ads
            self._scroll_to_load_ads()
            
            # Extract ad information
            ads_data = self._extract_ad_data(competitor_name)
            
            self.logger.info(f"Successfully scraped {len(ads_data)} ads for {competitor_name}")
            
        except Exception as e:
            self.logger.error(f"Error scraping ads for {competitor_name}: {e}")
        
        return ads_data
    
    def _scroll_to_load_ads(self):
        """Scroll down to load more ads"""
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        
        for _ in range(5):  # Scroll 5 times
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
    
    def _extract_ad_data(self, competitor_name):
        """Extract ad data from the current page"""
        ads_data = []
        
        try:
            # Find ad containers
            ad_elements = self.driver.find_elements(By.CSS_SELECTOR, "[data-testid='ad']")
            
            for ad_element in ad_elements[:FACEBOOK_SETTINGS["max_ads_per_competitor"]]:
                try:
                    ad_data = self._extract_single_ad(ad_element, competitor_name)
                    if ad_data:
                        ads_data.append(ad_data)
                except Exception as e:
                    self.logger.warning(f"Error extracting single ad: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error extracting ad data: {e}")
        
        return ads_data
    
    def _extract_single_ad(self, ad_element, competitor_name):
        """Extract data from a single ad element"""
        try:
            # Extract basic ad information
            ad_data = {
                "competitor": competitor_name,
                "scraped_at": datetime.now().isoformat(),
                "ad_type": "unknown",
                "content": "",
                "media_urls": [],
                "engagement_metrics": {},
                "targeting_info": {},
                "ad_copy": ""
            }
            
            # Try to extract ad copy
            try:
                text_elements = ad_element.find_elements(By.CSS_SELECTOR, "[data-testid='ad-text']")
                if text_elements:
                    ad_data["ad_copy"] = text_elements[0].text
            except:
                pass
            
            # Try to extract media URLs
            try:
                media_elements = ad_element.find_elements(By.CSS_SELECTOR, "img, video")
                for media in media_elements:
                    src = media.get_attribute("src")
                    if src:
                        ad_data["media_urls"].append(src)
            except:
                pass
            
            # Try to extract engagement metrics
            try:
                metric_elements = ad_element.find_elements(By.CSS_SELECTOR, "[data-testid*='metric']")
                for metric in metric_elements:
                    metric_text = metric.text
                    if "like" in metric_text.lower():
                        ad_data["engagement_metrics"]["likes"] = metric_text
                    elif "comment" in metric_text.lower():
                        ad_data["engagement_metrics"]["comments"] = metric_text
                    elif "share" in metric_text.lower():
                        ad_data["engagement_metrics"]["shares"] = metric_text
            except:
                pass
            
            return ad_data
            
        except Exception as e:
            self.logger.warning(f"Error extracting single ad data: {e}")
            return None
    
    def scrape_all_competitors(self):
        """Scrape ads for all configured competitors"""
        if not self.setup_driver():
            return {}
        
        all_ads_data = {}
        
        try:
            for competitor_name in COMPETITORS.keys():
                self.logger.info(f"Starting to scrape ads for {competitor_name}")
                ads_data = self.scrape_competitor_ads(competitor_name)
                all_ads_data[competitor_name] = ads_data
                
                # Delay between competitors
                time.sleep(FACEBOOK_SETTINGS["search_delay"] * 2)
            
            self.logger.info(f"Completed scraping for all competitors. Total ads collected: {sum(len(ads) for ads in all_ads_data.values())}")
            
        except Exception as e:
            self.logger.error(f"Error during scraping process: {e}")
        
        finally:
            if self.driver:
                self.driver.quit()
        
        return all_ads_data
    
    def save_ads_data(self, ads_data, filename=None):
        """Save scraped ads data to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"competitor_ads_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(ads_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Ads data saved to {filename}")
            return filename
        except Exception as e:
            self.logger.error(f"Error saving ads data: {e}")
            return None

def main():
    """Main function to run the scraper"""
    scraper = FacebookAdScraper()
    ads_data = scraper.scrape_all_competitors()
    
    if ads_data:
        scraper.save_ads_data(ads_data)
        print(f"Successfully scraped ads data for {len(ads_data)} competitors")
    else:
        print("No ads data was collected")

if __name__ == "__main__":
    main() 