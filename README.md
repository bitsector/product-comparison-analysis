# AI-Powered Product Comparison Analysis

🤖 **Smart Product Matching and Image Comparison Tools** that automatically match competitor products to your internal inventory using Google Gemini AI models for both text and image-based analysis.

## 📁 Project Files

### Python Scripts

#### 1. `product_matcher.py` - Text-Based Product Matching
- **Purpose**: Matches competitor products to your Fishbowl WMS inventory using text descriptions
- **AI Model**: Google Gemini text embeddings for semantic similarity
- **Input**: CSV files with competitor quotes and internal inventory
- **Output**: Detailed matching results with similarity scores and potential savings
- **Specialization**: Works with any product descriptions and text-based matching

#### 2. `image_comparison.py` - Visual Product Matching  
- **Purpose**: Compares competitor product images to your catalog images
- **AI Model**: Google Gemini Vision models for visual analysis
- **Input**: Image files in `their_images/` and `our_images/` directories
- **Output**: Visual similarity scores, side-by-side comparisons, and interchangeability analysis
- **Specialization**: Optimized for building materials (faucets, tiles, fixtures, hardware)

## ⚙️ Environment Configuration (.env)

### Required API Keys
```bash
# Google AI API Key (for Gemini models)
GOOGLE_AI_API_KEY=your_google_ai_api_key_here

# Legacy OpenAI key (if using OpenAI models)
OPENAI_API_KEY=your_openai_key_here
```

### Model Configuration
```bash
# Gemini model selection
GEMINI_MODEL=gemini-2.5-flash          # Current default
# Options: gemini-2.5-flash, gemini-2.0-flash, gemini-1.5-pro

# Image comparison method
IMAGE_COMPARISON_METHOD=vector         # "vector" (fast) or "prompt" (detailed)
```

### Application Settings
```bash
MATCH_CONFIDENCE_THRESHOLD=0.7         # Minimum confidence for matches
LOG_LEVEL=INFO                         # Logging verbosity
```

### Optional Fishbowl WMS Integration
```bash
FISHBOWL_SERVER_URL=http://localhost:28192
FISHBOWL_USERNAME=your_username
FISHBOWL_PASSWORD=your_password
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys
Edit `.env` file and add your Google AI API key:
```bash
GOOGLE_AI_API_KEY=your_actual_google_ai_api_key_here
```

### 3. Run Text-Based Product Matching
```bash
python product_matcher.py
```
- Processes CSV files in `mock_data/` directory
- Outputs results to `output/product_matching_results.csv`

### 4. Run Image-Based Product Comparison
```bash
python image_comparison.py
```
- Place competitor images in `their_images/` directory
- Place your catalog images in `our_images/` directory  
- Outputs results to `output/image_matching_results.csv`
- Opens visual comparisons in image viewer

## 🎯 How It Works

### Text-Based Matching (`product_matcher.py`)
1. **Load Data**: Reads competitor quotes and Fishbowl inventory from CSV files
2. **Generate Embeddings**: Creates semantic embeddings for product descriptions using Gemini
3. **Calculate Similarity**: Uses cosine similarity to find best matches
4. **Analyze Results**: Provides confidence scores, pricing comparisons, and stock availability
5. **Export Reports**: Generates comprehensive CSV reports with potential savings

### Image-Based Matching (`image_comparison.py`)
1. **Load Images**: Processes images from competitor and catalog directories
2. **AI Vision Analysis**: Uses Gemini Vision models to analyze product images
3. **Compare Products**: Evaluates functional equivalency and visual similarity
4. **Generate Reports**: Creates visual comparison displays and detailed analysis
5. **Cost Tracking**: Monitors API usage and calculates processing costs

### **Example Text Match:**
- **Competitor**: "Phillips screws 1/4" x 100pcs" 
- **Your Product**: "Phillips Head Screws 1/4 inch - Pack of 100"
- **AI Similarity**: 95% match confidence ✅
- **Savings**: $1.51 per unit

### **Example Image Match:**
- **Their Image**: Chrome bathroom faucet with dual handles
- **Your Image**: Chrome widespread bathroom faucet
- **Visual Similarity**: 87% match confidence
- **Interchangeable**: YES - functionally equivalent for construction projects

## 💰 Cost Estimates

### Google Gemini Pricing (2025)
- **Gemini 2.5 Flash**: $0.30/$2.50 per 1M tokens (input/output)
- **Gemini 2.0 Flash**: $0.10/$0.40 per 1M tokens (input/output)
- **Image Processing**: ~$0.0006-$0.0036 per image comparison
- **Text Matching**: ~$0.0003-$0.10 for typical catalogs

### Free Tier Available
- Google AI Studio provides free access with daily limits
- Perfect for development and testing

## 📊 Output Files

### Text Matching Results
- `output/product_matching_results.csv` - Complete matching analysis
- Includes similarity scores, pricing, stock levels, and savings calculations

### Image Matching Results  
- `output/image_matching_results.csv` - Image comparison summary
- `output/image_matching_results_detailed.json` - Full analysis with cost tracking
- `output/comparison_*.png` - Visual side-by-side comparisons
