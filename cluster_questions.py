import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import json
from collections import defaultdict

def load_data():
    """Load embeddings and similarity matrix."""
    # Load embeddings
    embedding_file = "question_embeddings.csv"
    if not os.path.exists(embedding_file):
        raise FileNotFoundError(f"Embedding file {embedding_file} not found.")
    
    df = pd.read_csv(embedding_file)
    question_ids = df['question_id'].values
    question_texts = df['question_text'].values
    
    # Extract embedding columns
    embedding_cols = [col for col in df.columns if col.startswith('embed_')]
    embeddings = df[embedding_cols].values
    
    # Load similarity matrix (optional, can be used for alternative clustering approaches)
    similarity_file = "question_similarity_matrix.csv"
    similarity_matrix = None
    if os.path.exists(similarity_file):
        similarity_df = pd.read_csv(similarity_file, index_col=0)
        similarity_matrix = similarity_df.values
    
    return question_ids, question_texts, embeddings, similarity_matrix

def perform_dbscan_clustering(embeddings, eps=0.3, min_samples=3):
    """Perform DBSCAN clustering on the embeddings."""
    print(f"Performing DBSCAN clustering with eps={eps}, min_samples={min_samples}...")
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    cluster_labels = dbscan.fit_predict(embeddings)
    
    # Count clusters (excluding noise points labeled as -1)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"Found {n_clusters} clusters and {n_noise} noise points")
    return cluster_labels

def visualize_clusters(embeddings, cluster_labels, question_ids, output_file):
    """Create a 2D visualization of the clusters using t-SNE."""
    # Reduce dimensionality for visualization
    print("Reducing dimensionality with t-SNE for visualization...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create scatter plot
    plt.figure(figsize=(12, 10))
    
    # Plot clustered points
    unique_labels = set(cluster_labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Black used for noise
            color = [0, 0, 0, 1]
        
        mask = cluster_labels == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=[color], label=f'Cluster {label}' if label != -1 else 'Noise',
                   alpha=0.7)
    
    # Add question IDs as labels
    for i, (x, y) in enumerate(embeddings_2d):
        plt.annotate(str(question_ids[i]), (x, y), fontsize=8)
    
    plt.title('Question Clusters Visualization')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Cluster visualization saved to {output_file}")

def organize_clusters(question_ids, question_texts, cluster_labels):
    """Organize questions by cluster."""
    clusters = defaultdict(list)
    
    for i, (qid, text, label) in enumerate(zip(question_ids, question_texts, cluster_labels)):
        cluster_name = f"Cluster {label}" if label != -1 else "Unclustered"
        clusters[cluster_name].append({
            "question_id": int(qid),
            "question_text": text
        })
    
    # Sort clusters by size (largest first)
    sorted_clusters = {k: clusters[k] for k in sorted(clusters.keys(), 
                                                     key=lambda x: len(clusters[x]), 
                                                     reverse=True)}
    
    # Sort questions within each cluster by question ID
    for cluster in sorted_clusters:
        sorted_clusters[cluster] = sorted(sorted_clusters[cluster], key=lambda x: x["question_id"])
    
    return sorted_clusters

def save_clusters(clusters, output_file):
    """Save clusters to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(clusters, f, indent=2)
    print(f"Clusters saved to {output_file}")

def print_cluster_summary(clusters):
    """Print a summary of the clusters."""
    print("\nCluster Summary:")
    for cluster_name, questions in clusters.items():
        print(f"\n{cluster_name} ({len(questions)} questions):")
        for q in questions[:5]:  # Print first 5 questions in each cluster
            print(f"  Question {q['question_id']}: {q['question_text']}")
        if len(questions) > 5:
            print(f"  ... and {len(questions) - 5} more questions")

def main():
    # Load data
    question_ids, question_texts, embeddings, similarity_matrix = load_data()
    
    # Try different DBSCAN parameters
    # These parameters may need tuning based on your specific data
    eps_values = [0.2, 0.3, 0.4]
    min_samples_values = [2, 3, 4]
    
    best_eps = 0.3
    best_min_samples = 3
    best_n_clusters = 0
    best_labels = None
    
    # Find parameters that give a reasonable number of clusters
    for eps in eps_values:
        for min_samples in min_samples_values:
            labels = perform_dbscan_clustering(embeddings, eps, min_samples)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            # Prefer parameters that give more clusters and fewer noise points
            # This is a simple heuristic - you might want to use a more sophisticated approach
            if n_clusters > best_n_clusters and n_noise < len(question_ids) * 0.3:
                best_eps = eps
                best_min_samples = min_samples
                best_n_clusters = n_clusters
                best_labels = labels
    
    print(f"\nBest parameters: eps={best_eps}, min_samples={best_min_samples}")
    print(f"Number of clusters: {best_n_clusters}")
    
    # If no good clustering was found, use the last one
    if best_labels is None:
        best_labels = labels
    
    # Visualize clusters
    visualize_clusters(embeddings, best_labels, question_ids, "question_clusters.png")
    
    # Organize and save clusters
    clusters = organize_clusters(question_ids, question_texts, best_labels)
    save_clusters(clusters, "question_clusters.json")
    
    # Print cluster summary
    print_cluster_summary(clusters)

if __name__ == "__main__":
    main() 