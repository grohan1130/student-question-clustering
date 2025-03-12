import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import re
import os

def load_processed_questions(file_path):
    """Load processed questions from file and return as a list of (id, text) tuples."""
    questions = []
    with open(file_path, 'r') as f:
        for line in f:
            match = re.match(r'(\d+)\.\s+(.*)', line.strip())
            if match:
                question_id = int(match.group(1))
                question_text = match.group(2)
                questions.append((question_id, question_text))
    return questions

def generate_embeddings(questions):
    """Generate embeddings for a list of questions using SBERT."""
    # Load the SBERT model
    print("Loading SBERT model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # A lightweight but effective model
    
    # Extract just the question texts for embedding
    question_texts = [q[1] for q in questions]
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = model.encode(question_texts, show_progress_bar=True)
    
    return embeddings

def save_embeddings(questions, embeddings, output_file):
    """Save question IDs, texts, and embeddings to a file."""
    # Create a DataFrame with question IDs and texts
    df = pd.DataFrame({
        'question_id': [q[0] for q in questions],
        'question_text': [q[1] for q in questions]
    })
    
    # Add embeddings as columns
    for i in range(embeddings.shape[1]):
        df[f'embed_{i}'] = embeddings[:, i]
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Embeddings saved to {output_file}")

def main():
    input_file = "linear_regression_questions_processed.txt"
    output_file = "question_embeddings.csv"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return
    
    # Load questions
    questions = load_processed_questions(input_file)
    print(f"Loaded {len(questions)} questions.")
    
    # Generate embeddings
    embeddings = generate_embeddings(questions)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Save embeddings
    save_embeddings(questions, embeddings, output_file)

if __name__ == "__main__":
    main() 