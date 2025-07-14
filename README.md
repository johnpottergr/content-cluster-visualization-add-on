# content-cluster-visualization-add-on

# LLM Content Cluster Visualization Add-On
Visualize website content as 3D topic clusters for SEO optimization.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Set `DEEPSEEK_API_KEY` in `.env`
3. Run: `python visualization.py --input urls.csv`

## Usage
- Input: CSV with `url` column (optional: `content`, `desired_topic`)
- Outputs: `content_visualization_output.csv`, `content_visualization.html`
