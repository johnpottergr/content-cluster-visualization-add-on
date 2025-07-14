import pandas as pd
import numpy as np
import umap
import networkx as nx
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import argparse
import os

# Initialize DeepSeek client
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "your_deepseek_api_key_here"),
    base_url="https://api.deepseek.com"
)

def scrape_content(url):
    """Scrape text content from a URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = " ".join(p.get_text(strip=True) for p in soup.find_all("p"))
        return text[:2000]  # Limit for efficiency
    except:
        return ""

def extract_keywords_and_embeddings(text, model_name='all-MiniLM-L6-v2'):
    """Generate embeddings using sentence-transformers."""
    model = SentenceTransformer(model_name)
    embedding = model.encode([text])[0]
    return embedding

def parse_embedding(s):
    """Parse embedding string from CSV (if provided)."""
    try:
        parts = [float(x) for x in s.split(",") if x]
        return np.array(parts)
    except:
        return None

def analyze_sentiment(text):
    """Integrate LLM Sentiment Analysis add-on (optional)."""
    prompt = f"Analyze sentiment of: {text[:500]}. Return positive, negative, or neutral."
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=10
        )
        return resp.choices[0].message.content.strip()
    except:
        return "neutral"

def generate_cluster_label(urls, contents):
    """Generate 3-5 word topic label using DeepSeek."""
    prompt = f"Summarize these URLs and their content into a 3-5 word topic label in Title Case: {', '.join(urls[:15])}"
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=10
        )
        return resp.choices[0].message.content.strip()
    except:
        return "Unnamed Cluster"

def visualize_content_clusters(input_csv="urls.csv", output_csv="content_visualization_output.csv", output_html="content_visualization.html", include_sentiment=False):
    """Generate 3D visualization of content topic clusters."""
    # Load CSV
    df = pd.read_csv(input_csv)
    print("Raw columns:", df.columns.tolist())

    # Validate columns
    required_cols = ["url"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns in {input_csv}. Found: {df.columns.tolist()}")

    # Scrape content or use provided content
    if "content" in df.columns:
        df["content"] = df["content"].fillna("")
    else:
        df["content"] = df["url"].apply(scrape_content)

    # Generate embeddings if not provided
    if "embedding" in df.columns:
        df["embedding"] = df["embedding"].apply(parse_embedding)
        df = df.dropna(subset=["embedding"])
    else:
        df["embedding"] = df["content"].apply(lambda x: extract_keywords_and_embeddings(x) if x else None)
        df = df.dropna(subset=["embedding"])
    if len(df) < 6:
        raise ValueError("Insufficient valid content/embeddings for clustering")
    embeddings = np.vstack(df["embedding"].values)
    print(f"Parsed embeddings: {len(df)}")

    # Add sentiment (optional)
    if include_sentiment:
        df["sentiment"] = df["content"].apply(lambda x: analyze_sentiment(x) if x else "neutral")
    else:
        df["sentiment"] = "neutral"

    # K-means clustering
    sil_scores = {}
    for k in range(6, min(20, len(df))):
        labels = KMeans(n_clusters=k, random_state=42).fit_predict(embeddings)
        sil_scores[k] = silhouette_score(embeddings, labels)
    best_k = max(sil_scores, key=sil_scores.get, default=6)
    df["cluster"] = KMeans(n_clusters=best_k, random_state=42).fit_predict(embeddings)
    print(f"Chosen clusters: {best_k}")

    # UMAP 3D reduction
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=min(15, len(df)-1),
        min_dist=0.1,
        metric="cosine",
        random_state=42
    )
    coords = reducer.fit_transform(embeddings)
    df[["x", "y", "z"]] = coords

    # Authority scoring
    centroid = embeddings.mean(axis=0, keepdims=True)
    df["authority_score"] = cosine_similarity(embeddings, centroid).flatten()
    cutoff = np.percentile(df["authority_score"], 1)
    df["off_topic"] = df["authority_score"] < cutoff
    print(f"Off-topic count: {df['off_topic'].sum()}")

    # Generate cluster labels
    cluster_topics = {}
    for c in sorted(df["cluster"].unique()):
        cluster_urls = df.loc[df["cluster"] == c, "url"].tolist()
        cluster_contents = df.loc[df["cluster"] == c, "content"].tolist()
        if cluster_urls:
            cluster_topics[c] = generate_cluster_label(cluster_urls, cluster_contents)
            print(f"Cluster {c}: {cluster_topics[c]}")
    df["cluster_topic"] = df["cluster"].map(cluster_topics)

    # Duplicate and link recommendations
    sim = cosine_similarity(embeddings)
    dup_thresh = 0.99
    link_low, link_high = 0.70, 0.99
    dup_recs, link_recs = [], []
    for i in range(len(df)):
        dups, links = [], []
        for j in range(len(df)):
            if i == j:
                continue
            s = sim[i, j]
            if s > dup_thresh:
                dups.append(df.at[j, "url"])
            elif link_low < s < link_high:
                links.append(df.at[j, "url"])
        dup_recs.append(dups[:3])
        link_recs.append(links[:5])
    for n in range(1, 4):
        df[f"potential_duplicate_content{n}"] = [rec[n-1] if len(rec) >= n else "" for rec in dup_recs]
    for n in range(1, 6):
        df[f"internal_link_opportunity{n}"] = [rec[n-1] if len(rec) >= n else "" for rec in link_recs]
    df["is_duplicate"] = [1 if rec else 0 for rec in dup_recs]

    # Graph-based duplicate refinement
    G = nx.Graph()
    G.add_nodes_from(df.index)
    for i, j in zip(*np.where(sim > dup_thresh)):
        if i < j and not df.at[i, "off_topic"] and not df.at[j, "off_topic"]:
            G.add_edge(i, j)
    for comp in nx.connected_components(G):
        if len(comp) > 1:
            ordered = sorted(comp, key=lambda i: df.at[i, "authority_score"], reverse=True)
            for idx in ordered[1:]:
                df.at[idx, "is_duplicate"] = 1

    # Categorize for plotting
    df["plot_cat"] = "central"
    df.loc[df["off_topic"], "plot_cat"] = "off_topic"
    df.loc[(~df["off_topic"]) & (df["is_duplicate"] == 1), "plot_cat"] = "duplicate"
    if include_sentiment:
        df.loc[df["sentiment"] == "positive", "plot_cat"] = "positive"
        df.loc[df["sentiment"] == "negative", "plot_cat"] = "negative"
        df.loc[df["sentiment"] == "neutral", "plot_cat"] = "neutral"

    # Generate 3D plot
    fig = go.Figure()
    colors = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#17becf", "#7f7f7f"]
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        col = colors[c % len(colors)]
        fig.add_trace(go.Scatter3d(
            x=sub.x, y=sub.y, z=sub.z,
            mode="markers",
            marker=dict(size=4, color=col),
            name=f"{cluster_topics[c]} ({len(sub)})",
            text=sub["url"],
            hovertemplate="URL: %{text}<br>Topic: " + sub["cluster_topic"] + "<extra></extra>"
        ))
    for cat, sym, nm in [
        ("positive", "circle", "Positive") if include_sentiment else (None, None, None),
        ("negative", "x", "Negative") if include_sentiment else (None, None, None),
        ("neutral", "square", "Neutral") if include_sentiment else (None, None, None),
        ("duplicate", "x", "Duplicates"),
        ("off_topic", "diamond", "Off-topic")
    ]:
        if cat:
            part = df[df["plot_cat"] == cat]
            if not part.empty:
                fig.add_trace(go.Scatter3d(
                    x=part.x, y=part.y, z=part.z,
                    mode="markers",
                    marker=dict(size=6, symbol=sym),
                    name=f"{nm} ({len(part)})",
                    text=part["url"],
                    hovertemplate="URL: %{text}<br>Topic: " + part["cluster_topic"] + "<extra></extra>"
                ))
    fig.update_layout(
        title="Content Topic Clustering",
        scene=dict(xaxis_title="UMAP1", yaxis_title="UMAP2", zaxis_title="UMAP3"),
        width=1200, height=600
    )
    fig.write_html(output_html)
    print(f"Saved visualization: {output_html}")

    # Save results
    output_cols = ["url", "cluster", "cluster_topic", "authority_score", "off_topic", "is_duplicate"] + \
                  [f"potential_duplicate_content{i}" for i in range(1, 4)] + \
                  [f"internal_link_opportunity{i}" for i in range(1, 6)] + \
                  (["sentiment"] if include_sentiment else [])
    df[output_cols].to_csv(output_csv, index=False)
    print(f"Saved CSV: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize content clusters for SEO.")
    parser.add_argument("--input", default="urls.csv", help="Input CSV with URLs")
    parser.add_argument("--output-csv", default="content_visualization_output.csv", help="Output CSV path")
    parser.add_argument("--output-html", default="content_visualization.html", help="Output HTML path")
    parser.add_argument("--sentiment", action="store_true", help="Include sentiment analysis")
    args = parser.parse_args()
    visualize_content_clusters(args.input, args.output_csv, args.output_html, args.sentiment)
