#!/usr/bin/env python3
"""
AI-Powered Product Matcher with Fishbowl WMS Integration
Matches competitor products to internal catalog using Google Gemini embeddings
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProductMatcher:
    """Main class for matching competitor products to Fishbowl inventory"""
    
    def __init__(self):
        """Initialize the ProductMatcher with Gemini client and configuration"""
        genai.configure(api_key=os.getenv('GOOGLE_AI_API_KEY'))
        self.confidence_threshold = float(os.getenv('MATCH_CONFIDENCE_THRESHOLD', 0.7))
        self.fishbowl_data = None
        self.competitor_data = None
        self.results = []
        
    def load_fishbowl_data(self, file_path: str = 'mock_data/fishbowl_inventory.csv') -> pd.DataFrame:
        """Load Fishbowl WMS inventory data from CSV file"""
        try:
            logger.info(f"Loading Fishbowl data from {file_path}")
            self.fishbowl_data = pd.read_csv(file_path)
            logger.info(f"Loaded {len(self.fishbowl_data)} Fishbowl products")
            return self.fishbowl_data
        except Exception as e:
            logger.error(f"Error loading Fishbowl data: {e}")
            raise
    
    def load_competitor_data(self, file_path: str = 'mock_data/competitor_quote.csv') -> pd.DataFrame:
        """Load competitor quote data from CSV file"""
        try:
            logger.info(f"Loading competitor data from {file_path}")
            self.competitor_data = pd.read_csv(file_path)
            logger.info(f"Loaded {len(self.competitor_data)} competitor products")
            return self.competitor_data
        except Exception as e:
            logger.error(f"Error loading competitor data: {e}")
            raise
    
    def get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding using Gemini's text embedding model"""
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="semantic_similarity"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error getting embedding for text: {text[:50]}... - {e}")
            return []
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two text descriptions"""
        try:
            embedding1 = self.get_text_embedding(text1)
            embedding2 = self.get_text_embedding(text2)
            
            if not embedding1 or not embedding2:
                return 0.0
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                [embedding1], 
                [embedding2]
            )[0][0]
            
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def find_best_matches(self, competitor_product: Dict) -> List[Dict]:
        """Find best matching Fishbowl products for a competitor product"""
        matches = []
        competitor_desc = competitor_product['product_name']
        
        logger.info(f"Finding matches for: {competitor_desc}")
        
        for _, fishbowl_product in self.fishbowl_data.iterrows():
            fishbowl_desc = fishbowl_product['description']
            
            # Calculate text similarity
            similarity_score = self.calculate_text_similarity(competitor_desc, fishbowl_desc)
            
            # Log detailed comparison with colored icons
            self._log_comparison_result(competitor_desc, fishbowl_desc, fishbowl_product['sku'], similarity_score)
            
            match_info = {
                'fishbowl_sku': fishbowl_product['sku'],
                'fishbowl_description': fishbowl_desc,
                'fishbowl_price': fishbowl_product['unit_price'],
                'fishbowl_stock': fishbowl_product['stock_quantity'],
                'fishbowl_category': fishbowl_product['category'],
                'fishbowl_supplier': fishbowl_product['supplier'],
                'fishbowl_image_url': fishbowl_product['image_url'],
                'similarity_score': similarity_score,
                'match_confidence': 'High' if similarity_score >= self.confidence_threshold else 'Low'
            }
            
            matches.append(match_info)
        
        # Sort by similarity score (highest first)
        matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return matches
    
    def _log_comparison_result(self, competitor_desc: str, fishbowl_desc: str, fishbowl_sku: str, similarity_score: float):
        """Log detailed comparison result with colored icons and text comparison"""
        # Determine match quality and icon
        if similarity_score >= 0.8:
            icon = "🟢"
            quality = "EXCELLENT MATCH"
        elif similarity_score >= self.confidence_threshold:
            icon = "🟡" 
            quality = "GOOD MATCH"
        elif similarity_score >= 0.4:
            icon = "🟠"
            quality = "FAIR MATCH"
        else:
            icon = "🔴"
            quality = "POOR MATCH"
        
        # Log the similarity score with icon
        logger.info(f"  ★ SIMILARITY SCORE: {similarity_score:.3f} | {icon} {quality}")
        
        # Log the text comparison
        logger.info(f"     COMPETITOR: \"{competitor_desc}\"")
        logger.info(f"     OUR PRODUCT: \"{fishbowl_desc}\" (SKU: {fishbowl_sku})")
        logger.info(f"     ---")
    
    def process_competitor_quote(self) -> List[Dict]:
        """Process entire competitor quote and find matches for each product"""
        if self.fishbowl_data is None or self.competitor_data is None:
            raise ValueError("Both Fishbowl and competitor data must be loaded first")
        
        logger.info("Starting product matching process...")
        self.results = []
        
        for _, competitor_product in self.competitor_data.iterrows():
            logger.info(f"Processing competitor SKU: {competitor_product['competitor_sku']}")
            
            # Find best matches
            matches = self.find_best_matches(competitor_product.to_dict())
            best_match = matches[0] if matches else None
            
            # Calculate extended pricing
            competitor_extended = competitor_product['quantity'] * competitor_product['competitor_price']
            our_extended = 0
            savings = 0
            
            if best_match:
                our_extended = competitor_product['quantity'] * best_match['fishbowl_price']
                savings = competitor_extended - our_extended
            
            result = {
                # Competitor info
                'competitor_sku': competitor_product['competitor_sku'],
                'competitor_product_name': competitor_product['product_name'],
                'competitor_quantity': competitor_product['quantity'],
                'competitor_unit_price': competitor_product['competitor_price'],
                'competitor_extended_price': competitor_extended,
                'competitor_image_url': competitor_product['product_image_url'],
                'competitor_notes': competitor_product.get('notes', ''),
                
                # Our best match info
                'our_sku': best_match['fishbowl_sku'] if best_match else 'NO MATCH',
                'our_description': best_match['fishbowl_description'] if best_match else '',
                'our_unit_price': best_match['fishbowl_price'] if best_match else 0,
                'our_extended_price': our_extended,
                'our_stock_quantity': best_match['fishbowl_stock'] if best_match else 0,
                'our_category': best_match['fishbowl_category'] if best_match else '',
                'our_supplier': best_match['fishbowl_supplier'] if best_match else '',
                'our_image_url': best_match['fishbowl_image_url'] if best_match else '',
                
                # Match analysis
                'match_confidence_score': best_match['similarity_score'] if best_match else 0,
                'match_confidence_level': best_match['match_confidence'] if best_match else 'No Match',
                'potential_savings': savings,
                'stock_availability': 'In Stock' if (best_match and best_match['fishbowl_stock'] >= competitor_product['quantity']) else 'Low Stock',
                'match_timestamp': datetime.now().isoformat()
            }
            
            self.results.append(result)
            
            # Log best match summary with colored icon
            if best_match:
                if result['match_confidence_score'] >= 0.8:
                    icon = "🟢"
                    quality = "EXCELLENT"
                elif result['match_confidence_score'] >= self.confidence_threshold:
                    icon = "🟡"
                    quality = "GOOD" 
                elif result['match_confidence_score'] >= 0.4:
                    icon = "🟠"
                    quality = "FAIR"
                else:
                    icon = "🔴"
                    quality = "POOR"
                
                logger.info(f"🎯 BEST MATCH for {competitor_product['competitor_sku']}: {result['our_sku']} | {icon} {quality} ({result['match_confidence_score']:.3f})")
                logger.info(f"   💰 Price: ${result['our_unit_price']:.2f} vs ${result['competitor_unit_price']:.2f} | Savings: ${result['potential_savings']:.2f}")
                logger.info(f"   📦 Stock: {result['our_stock_quantity']} units | Category: {result['our_category']}")
            else:
                logger.info(f"❌ NO MATCH FOUND for {competitor_product['competitor_sku']}")
        
        return self.results
    
    def export_results(self, output_file: str = 'output/product_matching_results.csv') -> str:
        """Export matching results to CSV file"""
        if not self.results:
            raise ValueError("No results to export. Run process_competitor_quote() first.")
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Convert results to DataFrame and export
            results_df = pd.DataFrame(self.results)
            results_df.to_csv(output_file, index=False)
            
            logger.info(f"Results exported to {output_file}")
            
            # Print summary
            total_products = len(self.results)
            high_confidence_matches = len([r for r in self.results if r['match_confidence_level'] == 'High'])
            total_savings = sum([r['potential_savings'] for r in self.results])
            
            logger.info(f"SUMMARY:")
            logger.info(f"- Total products processed: {total_products}")
            logger.info(f"- High confidence matches: {high_confidence_matches}")
            logger.info(f"- Total potential savings: ${total_savings:.2f}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            raise

def main():
    """Main function to run the product matcher"""
    logger.info("Starting AI-Powered Product Matcher (Gemini-powered)")
    
    try:
        # Initialize matcher
        matcher = ProductMatcher()
        
        # Load data
        matcher.load_fishbowl_data()
        matcher.load_competitor_data()
        
        # Process matching
        results = matcher.process_competitor_quote()
        
        # Print results summary with colored icons
        logger.info(f"Processed {len(results)} competitor products")
        for result in results:
            confidence = result['match_confidence_score']
            
            # Determine color icon based on confidence score
            if confidence >= 0.8:
                icon = "🟢"
            elif confidence >= self.confidence_threshold:
                icon = "🟡"
            elif confidence >= 0.4:
                icon = "🟠"
            else:
                icon = "🔴"
            
            logger.info(f"  {icon} {result['competitor_sku']} -> {result['our_sku']} (confidence: {confidence:.3f})")
        
        # Export results
        output_file = matcher.export_results()
        
        logger.info("Product matching completed successfully!")
        logger.info(f"Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        raise

if __name__ == "__main__":
    main()