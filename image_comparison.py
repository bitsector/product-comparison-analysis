#!/usr/bin/env python3
"""
AI-Powered Image Product Matcher with Fishbowl WMS Integration
Matches competitor product images to internal catalog using Google Gemini Vision models
Specialized for building materials: faucets, tiles, fixtures, hardware
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict
import logging
from dotenv import load_dotenv
import google.generativeai as genai
import json
from datetime import datetime
import glob
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import subprocess
import tempfile
import webbrowser

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ImageProductMatcher:
    """Main class for matching competitor product images to Fishbowl inventory"""
    
    def __init__(self):
        """Initialize the ImageProductMatcher with Gemini client and configuration"""
        genai.configure(api_key=os.getenv("GOOGLE_AI_API_KEY"))
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.client = genai.GenerativeModel(self.model)
        self.confidence_threshold = float(os.getenv("MATCH_CONFIDENCE_THRESHOLD", 0.7))
        self.competitor_images_dir = "their_images/"
        self.our_images_dir = "our_images/"
        self.results = []
        
        # Cost tracking
        self.total_api_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.model_pricing = self._get_model_pricing()
        
    def _get_model_pricing(self) -> Dict[str, Dict[str, float]]:
        """Get pricing information for different Gemini models (per 1M tokens)"""
        pricing = {
            "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
            "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
            "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
            "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
            "gemini-1.5-flash": {"input": 0.075, "output": 0.30}
        }
        return pricing.get(self.model, {"input": 0.30, "output": 2.50})
    
    def _track_usage(self, response) -> None:
        """Track API usage for cost calculation"""
        if hasattr(response, 'usage_metadata'):
            self.total_api_calls += 1
            self.total_input_tokens += response.usage_metadata.prompt_token_count
            self.total_output_tokens += response.usage_metadata.candidates_token_count
            
    def calculate_total_cost(self) -> Dict[str, float]:
        """Calculate total cost based on token usage"""
        input_cost = (self.total_input_tokens / 1_000_000) * self.model_pricing["input"]
        output_cost = (self.total_output_tokens / 1_000_000) * self.model_pricing["output"]
        total_cost = input_cost + output_cost
        
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "total_api_calls": self.total_api_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "model_used": self.model
        }
        
    def load_image_for_gemini(self, image_path: str):
        """Load image file for Gemini API"""
        try:
            import PIL.Image
            image = PIL.Image.open(image_path)
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def get_building_materials_comparison_prompt(self) -> str:
        """Get specialized prompt for building materials industry image comparison"""
        return """
        You are an expert in building materials, plumbing fixtures, tiles, and construction hardware. 

        Compare these two product images and determine if they represent the same or functionally 
        equivalent products for building/construction applications.

        ANALYZE THESE SPECIFIC ASPECTS:
        1. **Product Category**: Faucet, toilet, tile, hardware, fixture type
        2. **Functional Design**: Mounting type (deck/wall), handle configuration, operational mechanism  
        3. **Style & Finish**: Traditional/modern/contemporary, surface finish (chrome, brushed, matte, etc.)
        4. **Dimensions & Proportions**: Spout height, handle spacing, overall size relative to standard installations
        5. **Installation Type**: Widespread, centerset, single-hole, wall-mount, floor-mount
        6. **Material Quality Indicators**: Build quality, manufacturing grade (commercial vs residential)

        FOR FAUCETS - Focus on: Handle type (single/double), spout style (gooseneck/straight), 
                     spray features, mounting configuration
        FOR TOILETS - Focus on: One-piece vs two-piece, bowl shape (elongated/round), 
                     flush mechanism, mounting type  
        FOR TILES - Focus on: Size, pattern, texture, material type (ceramic/porcelain/natural stone), 
                   edge treatment
        FOR HARDWARE - Focus on: Mounting method, operational mechanism, load capacity, finish durability

        PROVIDE YOUR ASSESSMENT:
        - **Similarity Score**: 0.0 to 1.0 (where 1.0 = identical products from different manufacturers)
        - **Match Confidence**: HIGH (0.8-1.0) / MEDIUM (0.6-0.79) / LOW (0.4-0.59) / NO_MATCH (<0.4)  
        - **Product Match Type**: IDENTICAL / EQUIVALENT / SIMILAR_FUNCTION / DIFFERENT_CATEGORY
        - **Key Similarities**: List 3-5 specific matching features
        - **Key Differences**: List 3-5 specific differentiating factors
        - **Interchangeability**: Can these products serve the same function in a construction project? 
                                 (YES/NO with explanation)

        Focus on functional compatibility for building contractors and architects making product substitutions.
        """

    def resize_image_for_embedding(self, image_path: str, max_size: int = 256):
        """Resize image to fit within API limits (36KB) while maintaining aspect ratio"""
        import PIL.Image
        import io
        
        image = PIL.Image.open(image_path)
        original_size = image.size
        
        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Start with a smaller max size to ensure we stay under 36KB
        current_max_size = max_size
        
        while current_max_size > 64:  # Don't go too small
            # Calculate new size maintaining aspect ratio
            width, height = image.size
            if max(width, height) > current_max_size:
                if width > height:
                    new_width = current_max_size
                    new_height = int((height * current_max_size) / width)
                else:
                    new_height = current_max_size
                    new_width = int((width * current_max_size) / height)
                
                resized_image = image.resize((new_width, new_height), PIL.Image.Resampling.LANCZOS)
            else:
                resized_image = image
            
            # Check if the image size is under 36KB when saved as JPEG
            buffer = io.BytesIO()
            resized_image.save(buffer, format='JPEG', quality=85, optimize=True)
            size_bytes = buffer.tell()
            
            if size_bytes <= 35000:  # Leave some margin under 36KB
                logger.debug(f"Resized {image_path} from {original_size[0]}x{original_size[1]} to {resized_image.size[0]}x{resized_image.size[1]} ({size_bytes} bytes)")
                return resized_image
            
            # If still too large, reduce max size and try again
            current_max_size = int(current_max_size * 0.8)
        
        # Final fallback - very small image
        resized_image = image.resize((128, 128), PIL.Image.Resampling.LANCZOS)
        logger.warning(f"Had to resize {image_path} to very small size (128x128) to fit API limits")
        return resized_image

    def get_image_embedding(self, image_path: str) -> List[float]:
        """Get image description embedding using Gemini's vision + text embedding"""
        try:
            # Since multimodal embeddings are not available in Gemini API,
            # we'll use vision model to describe the image, then embed the description
            
            # Step 1: Get image description using vision model
            image = self.load_image_for_gemini(image_path)
            if image is None:
                return []
            
            description_prompt = """Describe this product image in detail for similarity matching. 
            Focus on: product type, style, finish, mounting type, key features, and functional aspects.
            Be specific and technical. Limit to 2-3 sentences."""
            
            response = self.client.generate_content([description_prompt, image])
            description = response.text
            
            # Step 2: Get text embedding of the description
            result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=description,
                task_type="semantic_similarity"
            )
            
            logger.debug(f"Image description for {image_path}: {description[:100]}...")
            return result['embedding']
            
        except Exception as e:
            logger.error(f"Error getting image embedding for {image_path}: {e}")
            return []
    
    def calculate_image_vector_similarity(self, their_image_path: str, our_image_path: str) -> float:
        """Calculate cosine similarity between two images using embeddings (fast method)"""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            their_embedding = self.get_image_embedding(their_image_path)
            our_embedding = self.get_image_embedding(our_image_path)
            
            if not their_embedding or not our_embedding:
                return 0.0
            
            similarity = cosine_similarity([their_embedding], [our_embedding])[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating image vector similarity: {e}")
            return 0.0

    def display_image_comparison(self, their_image_path: str, our_image_path: str, 
                               similarity_score: float, raw_analysis: str):
        """Display two images side by side with comparison score"""
        try:
            # Create figure with proper spacing and layout
            fig = plt.figure(figsize=(14, 8))
            
            # Create a grid layout with more control
            gs = fig.add_gridspec(2, 2, height_ratios=[1, 4], width_ratios=[1, 1], 
                                hspace=0.3, wspace=0.4, top=0.85, bottom=0.05, left=0.05, right=0.95)
            
            # Add main title at the top
            # Determine color and confidence level with normalized thresholds
            if similarity_score >= 0.95:
                color = 'green'
                confidence = 'EXCELLENT MATCH'
            elif similarity_score >= 0.80:
                color = 'orange'
                confidence = 'GOOD MATCH'
            else:
                color = 'red'
                confidence = 'POOR MATCH'
            
            # Add comparison result text at the very top
            fig.text(0.5, 0.95, f'{confidence} - Similarity Score: {similarity_score:.3f}', 
                    fontsize=18, fontweight='bold', color=color, ha='center', va='top')
            
            # Left image
            ax1 = fig.add_subplot(gs[1, 0])
            try:
                their_img = mpimg.imread(their_image_path)
                ax1.imshow(their_img)
            except Exception as e:
                ax1.text(0.5, 0.5, 'Image not available', ha='center', va='center', 
                        fontsize=14, transform=ax1.transAxes)
            ax1.axis('off')
            
            # Left image title - positioned above the image
            fig.text(0.25, 0.88, 'THEIR PRODUCT', fontsize=12, fontweight='bold', 
                    ha='center', va='center')
            fig.text(0.25, 0.85, f'{os.path.basename(their_image_path)}', fontsize=10, 
                    ha='center', va='center', style='italic')
            
            # Right image  
            ax2 = fig.add_subplot(gs[1, 1])
            try:
                our_img = mpimg.imread(our_image_path)
                ax2.imshow(our_img)
            except Exception as e:
                ax2.text(0.5, 0.5, 'Image not available', ha='center', va='center', 
                        fontsize=14, transform=ax2.transAxes)
            ax2.axis('off')
            
            # Right image title - positioned above the image
            fig.text(0.75, 0.88, 'OUR CATALOG PRODUCT', fontsize=12, fontweight='bold', 
                    ha='center', va='center')
            fig.text(0.75, 0.85, f'{os.path.basename(our_image_path)}', fontsize=10, 
                    ha='center', va='center', style='italic')
            
            # Save to file and open with system image viewer
            os.makedirs("output", exist_ok=True)
            timestamp = datetime.now().strftime("%H%M%S")
            output_file = f"output/comparison_{timestamp}_{similarity_score:.3f}.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            # Try multiple methods to open the image, preferring native viewers
            opened = False
            
            # Method 1: Try eog (Eye of GNOME - native Linux image viewer) with specific window size
            try:
                # eog will open in a reasonably sized window by default
                subprocess.Popen(['eog', output_file], 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                opened = True
                logger.info(f"🖼️  Opened comparison image in eog: {output_file}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
            
            # Method 2: Try other Linux image viewers
            if not opened:
                viewers = ['feh', 'gthumb', 'ristretto', 'gwenview', 'mirage']
                for viewer in viewers:
                    try:
                        subprocess.Popen([viewer, output_file],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        opened = True
                        logger.info(f"🖼️  Opened comparison image in {viewer}: {output_file}")
                        break
                    except FileNotFoundError:
                        continue
            
            # Method 3: Try xdg-open as fallback (this was opening Pinta)
            if not opened:
                try:
                    subprocess.run(['xdg-open', output_file], check=True, 
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    opened = True
                    logger.info(f"🖼️  Opened comparison image with xdg-open: {output_file}")
                except (subprocess.CalledProcessError, FileNotFoundError):
                    pass
            
            # Method 4: Try open (macOS) 
            if not opened:
                try:
                    subprocess.run(['open', output_file], check=True,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    opened = True
                    logger.info(f"🖼️  Opened comparison image: {output_file}")
                except (subprocess.CalledProcessError, FileNotFoundError):
                    pass
            
            # Method 5: Try webbrowser as last resort
            if not opened:
                try:
                    webbrowser.open(f'file://{os.path.abspath(output_file)}')
                    opened = True
                    logger.info(f"🖼️  Opened comparison image in browser: {output_file}")
                except:
                    pass
            
            if not opened:
                logger.info(f"💾 Comparison image saved: {output_file}")
                print(f"📸 Image comparison saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Error displaying image comparison: {e}")
            # Fallback to terminal display
            print(f"\n{'='*80}")
            print(f"🔍 IMAGE COMPARISON RESULTS")
            print(f"THEIR IMAGE: {their_image_path}")
            print(f"OUR IMAGE: {our_image_path}")
            print(f"SIMILARITY SCORE: {similarity_score:.3f}")
            print(f"{'='*80}\n")

    def compare_product_images(self, their_image_path: str, our_image_path: str) -> Dict:
        """Compare two product images using either prompt-based or vector-based method"""
        comparison_method = os.getenv("IMAGE_COMPARISON_METHOD", "prompt").lower()
        
        try:
            logger.info(f"Comparing images using {comparison_method} method")
            
            if comparison_method == "vector":
                # Fast vector-based comparison
                similarity_score = self.calculate_image_vector_similarity(their_image_path, our_image_path)
                
                return {
                    "similarity_score": similarity_score,
                    "match_confidence": "HIGH" if similarity_score >= 0.95 else "MEDIUM" if similarity_score >= 0.80 else "LOW",
                    "match_type": "VECTOR_SIMILARITY",
                    "key_similarities": ["Vector-based similarity analysis"],
                    "key_differences": ["Detailed analysis not available in vector mode"],
                    "interchangeability": "YES" if similarity_score >= 0.90 else "NO",
                    "raw_response": f"Vector similarity score: {similarity_score:.3f}"
                }
            
            else:
                # Detailed prompt-based comparison (original method)
                their_image = self.load_image_for_gemini(their_image_path)
                our_image = self.load_image_for_gemini(our_image_path)
                
                if their_image is None or our_image is None:
                    return {"error": "Failed to load images"}
                
                prompt = f"""{self.get_building_materials_comparison_prompt()}
                
                FIRST PRODUCT IMAGE:
                [Image will be provided]
                
                SECOND PRODUCT IMAGE:
                [Image will be provided]
                """
                
                response = self.client.generate_content([prompt, their_image, our_image])
                
                # Track usage for cost calculation
                self._track_usage(response)
                
                comparison_result = response.text
                
                # Parse the response to extract structured data
                parsed_result = self.parse_comparison_response(comparison_result)
                parsed_result["raw_response"] = comparison_result
                
                return parsed_result
            
        except Exception as e:
            logger.error(f"Error comparing images: {e}")
            return {"error": str(e), "similarity_score": 0.0}
    
    def parse_comparison_response(self, response_text: str) -> Dict:
        """Parse OpenAI response to extract structured comparison data"""
        try:
            # Initialize default values
            result = {
                "similarity_score": 0.0,
                "match_confidence": "LOW",
                "match_type": "DIFFERENT_CATEGORY", 
                "key_similarities": [],
                "key_differences": [],
                "interchangeability": "NO"
            }
            
            # Extract similarity score
            lines = response_text.split("\n")
            for line in lines:
                line = line.strip()
                if "similarity score" in line.lower() or "score" in line.lower():
                    # Look for decimal number in the line
                    import re
                    score_match = re.search(r"(\d+\.?\d*)", line)
                    if score_match:
                        score = float(score_match.group(1))
                        # Normalize if it's out of 0-1 range
                        if score > 1.0:
                            score = score / 10.0 if score <= 10 else score / 100.0
                        result["similarity_score"] = min(1.0, max(0.0, score))
                
                if "match confidence" in line.lower() or "confidence" in line.lower():
                    if "high" in line.lower():
                        result["match_confidence"] = "HIGH"
                    elif "medium" in line.lower():
                        result["match_confidence"] = "MEDIUM"
                    elif "low" in line.lower():
                        result["match_confidence"] = "LOW"
                
                if "interchangeability" in line.lower():
                    result["interchangeability"] = "YES" if "yes" in line.lower() else "NO"
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return {
                "similarity_score": 0.0,
                "match_confidence": "LOW", 
                "match_type": "PARSE_ERROR",
                "error": str(e)
            }
    
    def get_image_files(self, directory: str) -> List[str]:
        """Get all image files from a directory"""
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp"]
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(directory, ext)))
            image_files.extend(glob.glob(os.path.join(directory, ext.upper())))
        
        return sorted(image_files)
    
    def find_best_image_matches(self, their_image_path: str) -> List[Dict]:
        """Find best matching images from our catalog for their image"""
        our_images = self.get_image_files(self.our_images_dir)
        matches = []
        
        their_name = os.path.basename(their_image_path)
        logger.info(f"Finding matches for their image: {their_name}")
        
        for our_image_path in our_images:
            our_name = os.path.basename(our_image_path)
            logger.info(f"  Comparing with our image: {our_name}")
            
            comparison = self.compare_product_images(their_image_path, our_image_path)
            
            if "error" in comparison:
                logger.warning(f"  Error in comparison: {comparison['error']}")
                continue
            
            match_info = {
                "their_image": their_name,
                "our_image": our_name,
                "our_image_path": our_image_path,
                "similarity_score": comparison.get("similarity_score", 0.0),
                "match_confidence": comparison.get("match_confidence", "LOW"),
                "match_type": comparison.get("match_type", "UNKNOWN"),
                "interchangeability": comparison.get("interchangeability", "NO"),
                "key_similarities": comparison.get("key_similarities", []),
                "key_differences": comparison.get("key_differences", []),
                "raw_analysis": comparison.get("raw_response", ""),
                "comparison_timestamp": datetime.now().isoformat()
            }
            
            matches.append(match_info)
            
            # Enhanced logging with explicit similarity details
            score = match_info['similarity_score']
            confidence = match_info['match_confidence']
            interchangeable = match_info['interchangeability']
            
            logger.info(f"  ★ SIMILARITY SCORE: {score:.3f} ({confidence}) | Interchangeable: {interchangeable}")
            logger.info(f"    {their_name} ←→ {our_name}")
            
            # Adjusted thresholds for embedding similarity normalization
            if score >= 0.95:
                logger.info(f"    🟢 EXCELLENT MATCH - Very high similarity!")
            elif score >= 0.80:
                logger.info(f"    🟡 GOOD MATCH - Moderate similarity")
            else:
                logger.info(f"    🔴 POOR MATCH - Very low similarity")
                
            logger.info(f"    ---")
            
            # Display the comparison visually in browser
            self.display_image_comparison(
                their_image_path, 
                our_image_path, 
                score, 
                match_info.get('raw_analysis', 'No detailed analysis available')
            )
        
        # Sort by similarity score (highest first)
        matches.sort(key=lambda x: x["similarity_score"], reverse=True)
        return matches
    
    def process_all_their_images(self) -> List[Dict]:
        """Process all their images and find matches"""
        their_images = self.get_image_files(self.competitor_images_dir)
        
        if not their_images:
            raise ValueError(f"No images found in {self.competitor_images_dir}")
        
        logger.info(f"Found {len(their_images)} images to process")
        
        self.results = []
        
        for their_image_path in their_images:
            matches = self.find_best_image_matches(their_image_path)
            
            # Get the best match
            best_match = matches[0] if matches else None
            
            result = {
                "their_image": os.path.basename(their_image_path),
                "their_image_path": their_image_path,
                "best_match_image": best_match["our_image"] if best_match else "NO_MATCH",
                "best_match_path": best_match["our_image_path"] if best_match else "",
                "similarity_score": best_match["similarity_score"] if best_match else 0.0,
                "match_confidence": best_match["match_confidence"] if best_match else "NO_MATCH",
                "match_type": best_match["match_type"] if best_match else "NO_MATCH",
                "interchangeable": best_match["interchangeability"] if best_match else "NO",
                "analysis_summary": best_match["raw_analysis"] if best_match else "No matches found",
                "all_matches": matches,
                "processed_timestamp": datetime.now().isoformat()
            }
            
            self.results.append(result)
            
            logger.info(f"Best match for {result['their_image']}: {result['best_match_image']} (score: {result['similarity_score']:.3f})")
        
        return self.results
    
    def export_results(self, output_file: str = "output/image_matching_results.csv") -> str:
        """Export image matching results to CSV file"""
        if not self.results:
            raise ValueError("No results to export. Run process_all_competitor_images() first.")
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Prepare data for CSV (flatten the results)
            csv_data = []
            for result in self.results:
                csv_row = {
                    "their_image": result["their_image"],
                    "best_match_image": result["best_match_image"], 
                    "similarity_score": result["similarity_score"],
                    "match_confidence": result["match_confidence"],
                    "match_type": result["match_type"],
                    "interchangeable": result["interchangeable"],
                    "processed_timestamp": result["processed_timestamp"]
                }
                csv_data.append(csv_row)
            
            # Export to CSV
            results_df = pd.DataFrame(csv_data)
            results_df.to_csv(output_file, index=False)
            
            logger.info(f"Results exported to {output_file}")
            
            # Print summary
            total_images = len(self.results)
            high_confidence = len([r for r in self.results if r["match_confidence"] == "HIGH"])
            interchangeable = len([r for r in self.results if r["interchangeable"] == "YES"])
            avg_score = np.mean([r["similarity_score"] for r in self.results])
            
            # Calculate and display cost summary
            cost_summary = self.calculate_total_cost()
            
            logger.info("ANALYSIS SUMMARY:")
            logger.info(f"- Total images processed: {total_images}")
            logger.info(f"- High confidence matches: {high_confidence}")
            logger.info(f"- Interchangeable products found: {interchangeable}")  
            logger.info(f"- Average similarity score: {avg_score:.3f}")
            
            logger.info("\nCOST SUMMARY:")
            logger.info(f"- Model used: {cost_summary['model_used']}")
            logger.info(f"- Total API calls: {cost_summary['total_api_calls']}")
            logger.info(f"- Input tokens: {cost_summary['total_input_tokens']:,}")
            logger.info(f"- Output tokens: {cost_summary['total_output_tokens']:,}")
            logger.info(f"- Input cost: ${cost_summary['input_cost']:.4f}")
            logger.info(f"- Output cost: ${cost_summary['output_cost']:.4f}")
            logger.info(f"- TOTAL COST: ${cost_summary['total_cost']:.4f}")
            if total_images > 0:
                cost_per_image = cost_summary['total_cost'] / total_images
                logger.info(f"- Average cost per image: ${cost_per_image:.4f}")
            
            # Add cost summary to JSON export
            cost_summary_data = {
                "cost_analysis": cost_summary,
                "analysis_summary": {
                    "total_images": total_images,
                    "high_confidence_matches": high_confidence,
                    "interchangeable_products": interchangeable,
                    "average_similarity_score": avg_score,
                    "cost_per_image": cost_summary['total_cost'] / total_images if total_images > 0 else 0
                }
            }
            
            # Export detailed JSON results with cost data
            json_file = output_file.replace(".csv", "_detailed.json")
            export_data = {
                "results": self.results,
                "summary": cost_summary_data
            }
            with open(json_file, "w") as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Detailed results with cost analysis exported to {json_file}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            raise

def main():
    """Main function to run the image product matcher"""
    logger.info("Starting AI-Powered Image Product Matcher for Building Materials (Gemini-powered)")
    
    try:
        # Initialize matcher
        matcher = ImageProductMatcher()
        
        # Check directories exist
        if not os.path.exists(matcher.competitor_images_dir):
            os.makedirs(matcher.competitor_images_dir)
            logger.warning(f"Created {matcher.competitor_images_dir} - please add their product images")
        
        if not os.path.exists(matcher.our_images_dir):
            os.makedirs(matcher.our_images_dir)
            logger.warning(f"Created {matcher.our_images_dir} - please add your catalog images")
        
        # Process all images
        results = matcher.process_all_their_images()
        
        # Export results
        output_file = matcher.export_results()
        
        logger.info("Image product matching completed successfully!")
        logger.info(f"Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        raise

if __name__ == "__main__":
    main()