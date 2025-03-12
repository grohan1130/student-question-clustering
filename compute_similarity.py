import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_embeddings(file_path):
    """Load question embeddings from CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Embedding file {file_path} not found.")
    
    df = pd.read_csv(file_path)
    
    # Extract question IDs and texts
    question_ids = df['question_id'].values
    question_texts = df['question_text'].values
    
    # Extract embedding columns (all columns starting with 'embed_')
    embedding_cols = [col for col in df.columns if col.startswith('embed_')]
    embeddings = df[embedding_cols].values
    
    print(f"Loaded {len(question_ids)} questions with embeddings of dimension {embeddings.shape[1]}")
    return question_ids, question_texts, embeddings

def compute_similarity_matrix(embeddings):
    """Compute cosine similarity between all pairs of embeddings."""
    print("Computing cosine similarity matrix...")
    similarity_matrix = cosine_similarity(embeddings)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    return similarity_matrix

def save_similarity_matrix(similarity_matrix, question_ids, output_file):
    """Save similarity matrix to CSV file."""
    # Create DataFrame with question IDs as index and columns
    df = pd.DataFrame(similarity_matrix, 
                      index=question_ids, 
                      columns=question_ids)
    
    # Save to CSV
    df.to_csv(output_file)
    print(f"Similarity matrix saved to {output_file}")
    return df

def visualize_similarity_matrix(similarity_matrix, question_ids, output_file):
    """Create a heatmap visualization of the similarity matrix."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, cmap='viridis', xticklabels=False, yticklabels=False)
    plt.title('Question Similarity Matrix (Cosine Similarity)')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Similarity matrix visualization saved to {output_file}")

def main():
    embedding_file = "question_embeddings.csv"
    similarity_output_file = "question_similarity_matrix.csv"
    visualization_output_file = "similarity_heatmap.png"
    
    # Load embeddings
    question_ids, question_texts, embeddings = load_embeddings(embedding_file)
    
    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(embeddings)
    
    # Save similarity matrix
    similarity_df = save_similarity_matrix(similarity_matrix, question_ids, similarity_output_file)
    
    # Visualize similarity matrix
    visualize_similarity_matrix(similarity_matrix, question_ids, visualization_output_file)
    
    # Print some example similarities
    print("\nExample similarities between questions:")
    for i in range(min(5, len(question_ids))):
        most_similar_idx = np.argsort(similarity_matrix[i])[::-1][1:4]  # Top 3 most similar (excluding self)
        print(f"\nQuestion {question_ids[i]}: {question_texts[i]}")
        print("Most similar questions:")
        for idx in most_similar_idx:
            similarity = similarity_matrix[i, idx]
            print(f"  Question {question_ids[idx]} (similarity: {similarity:.3f}): {question_texts[idx]}")

if __name__ == "__main__":
    main() 