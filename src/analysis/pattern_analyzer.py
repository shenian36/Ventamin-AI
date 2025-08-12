"""
Pattern Analyzer for Competitor Ad Analysis
Identifies successful patterns and strategies from competitor advertisements
"""

import json
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import re
from datetime import datetime
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

from config import ANALYSIS_SETTINGS

class PatternAnalyzer:
    def __init__(self):
        self.setup_logging()
        self.ads_data = {}
        self.analysis_results = {}
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_ads_data(self, filename):
        """Load ads data from JSON file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.ads_data = json.load(f)
            self.logger.info(f"Loaded ads data from {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading ads data: {e}")
            return False
    
    def analyze_text_patterns(self):
        """Analyze text patterns in ad copy"""
        all_ad_copies = []
        competitor_ad_copies = defaultdict(list)
        
        for competitor, ads in self.ads_data.items():
            for ad in ads:
                if ad.get('ad_copy'):
                    all_ad_copies.append(ad['ad_copy'])
                    competitor_ad_copies[competitor].append(ad['ad_copy'])
        
        # Text analysis results
        text_analysis = {
            'common_phrases': self._extract_common_phrases(all_ad_copies),
            'competitor_specific_phrases': self._analyze_competitor_phrases(competitor_ad_copies),
            'sentiment_analysis': self._analyze_sentiment(all_ad_copies),
            'call_to_action_patterns': self._extract_cta_patterns(all_ad_copies)
        }
        
        return text_analysis
    
    def _extract_common_phrases(self, ad_copies):
        """Extract common phrases from ad copies"""
        # Clean and tokenize text
        cleaned_texts = []
        for text in ad_copies:
            # Remove special characters and convert to lowercase
            cleaned = re.sub(r'[^\w\s]', '', text.lower())
            cleaned_texts.append(cleaned)
        
        # Use TF-IDF to find important phrases
        vectorizer = TfidfVectorizer(
            ngram_range=(2, 4),
            min_df=2,
            max_df=0.8,
            stop_words='english'
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top phrases by TF-IDF score
            tfidf_scores = np.sum(tfidf_matrix.toarray(), axis=0)
            top_indices = np.argsort(tfidf_scores)[-20:]  # Top 20 phrases
            
            common_phrases = [feature_names[i] for i in top_indices]
            return common_phrases
            
        except Exception as e:
            self.logger.warning(f"Error in phrase extraction: {e}")
            return []
    
    def _analyze_competitor_phrases(self, competitor_ad_copies):
        """Analyze phrases specific to each competitor"""
        competitor_phrases = {}
        
        for competitor, ad_copies in competitor_ad_copies.items():
            if not ad_copies:
                continue
                
            # Clean texts
            cleaned_texts = []
            for text in ad_copies:
                cleaned = re.sub(r'[^\w\s]', '', text.lower())
                cleaned_texts.append(cleaned)
            
            # Extract phrases for this competitor
            vectorizer = TfidfVectorizer(
                ngram_range=(2, 3),
                min_df=1,
                max_df=0.9
            )
            
            try:
                tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
                feature_names = vectorizer.get_feature_names_out()
                
                # Get top phrases
                tfidf_scores = np.sum(tfidf_matrix.toarray(), axis=0)
                top_indices = np.argsort(tfidf_scores)[-10:]  # Top 10 phrases
                
                competitor_phrases[competitor] = [feature_names[i] for i in top_indices]
                
            except Exception as e:
                self.logger.warning(f"Error analyzing phrases for {competitor}: {e}")
                competitor_phrases[competitor] = []
        
        return competitor_phrases
    
    def _analyze_sentiment(self, ad_copies):
        """Basic sentiment analysis of ad copies"""
        positive_words = ['amazing', 'best', 'great', 'excellent', 'perfect', 'wonderful', 'fantastic', 'incredible']
        negative_words = ['worst', 'terrible', 'awful', 'bad', 'poor', 'disappointing', 'horrible']
        
        sentiment_results = {
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'positive_phrases': [],
            'negative_phrases': []
        }
        
        for text in ad_copies:
            text_lower = text.lower()
            positive_matches = sum(1 for word in positive_words if word in text_lower)
            negative_matches = sum(1 for word in negative_words if word in text_lower)
            
            if positive_matches > negative_matches:
                sentiment_results['positive_count'] += 1
            elif negative_matches > positive_matches:
                sentiment_results['negative_count'] += 1
            else:
                sentiment_results['neutral_count'] += 1
        
        return sentiment_results
    
    def _extract_cta_patterns(self, ad_copies):
        """Extract call-to-action patterns"""
        cta_patterns = {
            'common_ctas': [],
            'cta_frequency': {}
        }
        
        cta_keywords = [
            'buy now', 'shop now', 'learn more', 'get started', 'sign up',
            'order now', 'try now', 'discover', 'explore', 'click here',
            'limited time', 'offer', 'discount', 'sale', 'free'
        ]
        
        for text in ad_copies:
            text_lower = text.lower()
            for cta in cta_keywords:
                if cta in text_lower:
                    cta_patterns['cta_frequency'][cta] = cta_patterns['cta_frequency'].get(cta, 0) + 1
        
        # Get most common CTAs
        sorted_ctas = sorted(cta_patterns['cta_frequency'].items(), key=lambda x: x[1], reverse=True)
        cta_patterns['common_ctas'] = [cta for cta, count in sorted_ctas[:10]]
        
        return cta_patterns
    
    def analyze_visual_patterns(self):
        """Analyze visual patterns in ads"""
        visual_analysis = {
            'ad_types': Counter(),
            'media_types': Counter(),
            'color_patterns': {},
            'layout_patterns': {}
        }
        
        for competitor, ads in self.ads_data.items():
            for ad in ads:
                # Analyze ad types
                ad_type = ad.get('ad_type', 'unknown')
                visual_analysis['ad_types'][ad_type] += 1
                
                # Analyze media types
                media_urls = ad.get('media_urls', [])
                for url in media_urls:
                    if url.endswith(('.jpg', '.jpeg', '.png')):
                        visual_analysis['media_types']['image'] += 1
                    elif url.endswith(('.mp4', '.mov', '.avi')):
                        visual_analysis['media_types']['video'] += 1
        
        return visual_analysis
    
    def analyze_engagement_patterns(self):
        """Analyze engagement patterns"""
        engagement_analysis = {
            'high_performing_ads': [],
            'engagement_metrics': {},
            'competitor_performance': {}
        }
        
        for competitor, ads in self.ads_data.items():
            competitor_metrics = {
                'total_ads': len(ads),
                'avg_likes': 0,
                'avg_comments': 0,
                'avg_shares': 0
            }
            
            total_likes = 0
            total_comments = 0
            total_shares = 0
            
            for ad in ads:
                metrics = ad.get('engagement_metrics', {})
                
                # Extract numeric values from engagement metrics
                likes = self._extract_numeric_value(metrics.get('likes', '0'))
                comments = self._extract_numeric_value(metrics.get('comments', '0'))
                shares = self._extract_numeric_value(metrics.get('shares', '0'))
                
                total_likes += likes
                total_comments += comments
                total_shares += shares
                
                # Identify high-performing ads
                total_engagement = likes + comments + shares
                if total_engagement > 100:  # Threshold for high-performing
                    engagement_analysis['high_performing_ads'].append({
                        'competitor': competitor,
                        'ad_copy': ad.get('ad_copy', ''),
                        'engagement': total_engagement,
                        'metrics': metrics
                    })
            
            # Calculate averages
            if len(ads) > 0:
                competitor_metrics['avg_likes'] = total_likes / len(ads)
                competitor_metrics['avg_comments'] = total_comments / len(ads)
                competitor_metrics['avg_shares'] = total_shares / len(ads)
            
            engagement_analysis['competitor_performance'][competitor] = competitor_metrics
        
        return engagement_analysis
    
    def _extract_numeric_value(self, text):
        """Extract numeric value from text (e.g., '1.2K' -> 1200)"""
        if not text:
            return 0
        
        # Remove non-numeric characters except K, M, B
        cleaned = re.sub(r'[^\d.KMB]', '', text.upper())
        
        try:
            if 'K' in cleaned:
                return int(float(cleaned.replace('K', '')) * 1000)
            elif 'M' in cleaned:
                return int(float(cleaned.replace('M', '')) * 1000000)
            elif 'B' in cleaned:
                return int(float(cleaned.replace('B', '')) * 1000000000)
            else:
                return int(float(cleaned))
        except:
            return 0
    
    def cluster_similar_ads(self):
        """Cluster similar ads using text similarity"""
        ad_texts = []
        ad_metadata = []
        
        for competitor, ads in self.ads_data.items():
            for ad in ads:
                if ad.get('ad_copy'):
                    ad_texts.append(ad['ad_copy'])
                    ad_metadata.append({
                        'competitor': competitor,
                        'ad_copy': ad['ad_copy'],
                        'engagement': ad.get('engagement_metrics', {})
                    })
        
        if len(ad_texts) < 2:
            return []
        
        # Vectorize text
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            min_df=1,
            max_df=0.9
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(ad_texts)
            
            # Perform clustering
            n_clusters = min(5, len(ad_texts))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(tfidf_matrix)
            
            # Group ads by cluster
            clustered_ads = defaultdict(list)
            for i, cluster_id in enumerate(clusters):
                clustered_ads[f"cluster_{cluster_id}"].append(ad_metadata[i])
            
            return dict(clustered_ads)
            
        except Exception as e:
            self.logger.warning(f"Error in ad clustering: {e}")
            return []
    
    def generate_insights(self):
        """Generate comprehensive insights from analysis"""
        insights = {
            'text_patterns': self.analyze_text_patterns(),
            'visual_patterns': self.analyze_visual_patterns(),
            'engagement_patterns': self.analyze_engagement_patterns(),
            'similar_ads': self.cluster_similar_ads(),
            'recommendations': self._generate_recommendations()
        }
        
        self.analysis_results = insights
        return insights
    
    def _generate_recommendations(self):
        """Generate recommendations based on analysis"""
        recommendations = {
            'content_strategy': [],
            'visual_strategy': [],
            'engagement_strategy': []
        }
        
        # Content recommendations based on text analysis
        text_analysis = self.analyze_text_patterns()
        if text_analysis['common_phrases']:
            recommendations['content_strategy'].append(
                f"Incorporate common phrases: {', '.join(text_analysis['common_phrases'][:5])}"
            )
        
        if text_analysis['call_to_action_patterns']['common_ctas']:
            recommendations['content_strategy'].append(
                f"Use effective CTAs: {', '.join(text_analysis['call_to_action_patterns']['common_ctas'][:3])}"
            )
        
        # Visual recommendations
        visual_analysis = self.analyze_visual_patterns()
        if visual_analysis['media_types']:
            most_common_media = visual_analysis['media_types'].most_common(1)[0]
            recommendations['visual_strategy'].append(
                f"Focus on {most_common_media[0]} content as it's most common"
            )
        
        # Engagement recommendations
        engagement_analysis = self.analyze_engagement_patterns()
        if engagement_analysis['high_performing_ads']:
            recommendations['engagement_strategy'].append(
                f"Study {len(engagement_analysis['high_performing_ads'])} high-performing ads for patterns"
            )
        
        return recommendations
    
    def save_analysis(self, filename=None):
        """Save analysis results to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pattern_analysis_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Analysis results saved to {filename}")
            return filename
        except Exception as e:
            self.logger.error(f"Error saving analysis: {e}")
            return None

def main():
    """Main function to run pattern analysis"""
    analyzer = PatternAnalyzer()
    
    # Load sample data or use actual data file
    sample_data = {
        "G1": [
            {"ad_copy": "Transform your health with our premium supplements!", "engagement_metrics": {"likes": "1.2K"}},
            {"ad_copy": "Get the best wellness products today!", "engagement_metrics": {"likes": "800"}}
        ],
        "M8": [
            {"ad_copy": "Discover amazing health benefits now!", "engagement_metrics": {"likes": "950"}},
            {"ad_copy": "Try our revolutionary supplements!", "engagement_metrics": {"likes": "1.5K"}}
        ]
    }
    
    analyzer.ads_data = sample_data
    insights = analyzer.generate_insights()
    
    print("Pattern Analysis Complete!")
    print(f"Found {len(insights['text_patterns']['common_phrases'])} common phrases")
    print(f"Identified {len(insights['engagement_patterns']['high_performing_ads'])} high-performing ads")
    
    analyzer.save_analysis()

if __name__ == "__main__":
    main() 