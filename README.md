# AI-Powered Product Matcher with Fishbowl WMS Integration

🤖 **Smart Product Matching Tool** that automatically matches competitor products to your internal inventory using OpenAI's text embeddings and semantic similarity analysis. 

## 🎯 What This Project Does

This tool helps businesses **generate competitive quotes faster** by:

1. **Reading competitor quotes** (CSV with product descriptions)
2. **Loading your Fishbowl WMS inventory** (or mock data for testing)
3. **Using AI to match products** based on text similarity (not just exact names)
4. **Calculating similarity scores** and confidence levels for each match
5. **Generating comparison reports** showing potential savings and stock availability

### **Example Match:**
- **Competitor**: "Phillips screws 1/4" x 100pcs" 
- **Your Product**: "Phillips Head Screws 1/4 inch - Pack of 100"
- **AI Similarity**: 95% match confidence ✅
- **Savings**: $1.51 per unit

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (without CUDA - CPU only)
pip install -r requirements.txt
```

### 2. Configure API Key
Edit `.env` file and add your OpenAI API key:
```bash
OPENAI_API_KEY=your_actual_openai_api_key_here
```

### 3. Run the Matcher
```bash
python product_matcher.py
```

### 4. Check Results
- **Console**: See real-time matching progress and confidence scores
- **Output File**: `output/product_matching_results.csv` with detailed comparisons

## 💰 Cost Estimate
- **Current test data**: ~$0.0003 (less than 1 cent)
- **Typical business use**: $0.005 - $0.10 for most catalogs
- Uses **text-embedding-3-small** model ($0.02 per 1M tokens)
