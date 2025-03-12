import json
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import openai
import time
import re
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_clusters(file_path="question_clusters.json"):
    """Load clusters from JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cluster file {file_path} not found.")
    
    with open(file_path, 'r') as f:
        clusters = json.load(f)
    
    print(f"Loaded {len(clusters)} clusters")
    return clusters

def load_original_questions(file_path="linear_regression_questions_raw.txt"):
    """Load original questions for better labeling."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Original questions file {file_path} not found.")
    
    questions = {}
    with open(file_path, 'r') as f:
        for line in f:
            match = re.match(r'(\d+)\.\s+(.*)', line.strip())
            if match:
                question_id = int(match.group(1))
                question_text = match.group(2)
                questions[question_id] = question_text
    
    print(f"Loaded {len(questions)} original questions")
    return questions

def extract_tfidf_keywords(cluster_texts, n_keywords=5):
    """Extract top keywords from cluster using TF-IDF."""
    # Skip if cluster is empty
    if not cluster_texts:
        return []
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        ngram_range=(1, 2)  # Include both unigrams and bigrams
    )
    
    # Fit and transform the texts
    tfidf_matrix = vectorizer.fit_transform(cluster_texts)
    
    # Get feature names (terms)
    feature_names = vectorizer.get_feature_names_out()
    
    # Calculate average TF-IDF score for each term across all documents
    avg_tfidf_scores = np.mean(tfidf_matrix.toarray(), axis=0)
    
    # Get indices of top terms
    top_indices = avg_tfidf_scores.argsort()[-n_keywords:][::-1]
    
    # Get top terms
    top_terms = [feature_names[i] for i in top_indices]
    
    return top_terms

def generate_llm_label(cluster_questions, keywords):
    """Generate a descriptive label for the cluster using an LLM."""
    try:
        # Prepare prompt with questions and keywords
        prompt = f"""
        I have a cluster of questions about linear regression. Please generate a concise label (5-7 words) 
        that captures the main theme of these questions. The label should be specific enough to distinguish 
        this cluster from other linear regression topics.
        
        Questions in the cluster:
        {chr(10).join([f"- {q}" for q in cluster_questions[:10]])}
        
        Keywords extracted from these questions:
        {', '.join(keywords)}
        
        Label:
        """
        
        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates concise, descriptive labels for groups of related questions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=20,
            temperature=0.3
        )
        
        # Extract and clean the label
        label = response.choices[0].message.content.strip()
        # Remove quotes if present
        label = re.sub(r'^["\'](.*)["\']$', r'\1', label)
        
        return label
    
    except Exception as e:
        print(f"Error generating LLM label: {e}")
        # Fallback to keywords if LLM fails
        return f"Questions about {', '.join(keywords[:3])}"

def generate_fallback_label(cluster_name, keywords):
    """Generate a fallback label using keywords when LLM is not available."""
    if not keywords:
        return f"{cluster_name} (Miscellaneous Questions)"
    
    # Join top 3 keywords
    keyword_text = ', '.join(keywords[:3])
    return f"Questions about {keyword_text}"

def label_clusters(clusters, original_questions, use_llm=False):
    """Generate labels for each cluster."""
    labeled_clusters = {}
    
    for cluster_name, questions in clusters.items():
        # Skip empty clusters
        if not questions:
            continue
        
        # Get original question texts
        cluster_question_texts = [original_questions.get(q["question_id"], q["question_text"]) 
                                 for q in questions]
        
        # Extract keywords using TF-IDF
        keywords = extract_tfidf_keywords(cluster_question_texts)
        
        # Generate label
        if use_llm and openai.api_key:
            try:
                label = generate_llm_label(cluster_question_texts, keywords)
                # Add a small delay to avoid rate limits
                time.sleep(1)
            except Exception as e:
                print(f"Error with LLM, falling back to keyword-based label: {e}")
                label = generate_fallback_label(cluster_name, keywords)
        else:
            label = generate_fallback_label(cluster_name, keywords)
        
        # Store labeled cluster
        labeled_clusters[label] = {
            "original_name": cluster_name,
            "keywords": keywords,
            "questions": questions
        }
    
    return labeled_clusters

def save_labeled_clusters(labeled_clusters, output_file="labeled_clusters.json"):
    """Save labeled clusters to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(labeled_clusters, f, indent=2)
    print(f"Labeled clusters saved to {output_file}")

def generate_readable_output(labeled_clusters, original_questions, output_file="categorized_questions.txt"):
    """Generate a human-readable text file with categorized questions."""
    with open(output_file, 'w') as f:
        f.write("# CATEGORIZED LINEAR REGRESSION QUESTIONS\n\n")
        
        for label, cluster_data in labeled_clusters.items():
            f.write(f"## {label}\n")
            f.write(f"Keywords: {', '.join(cluster_data['keywords'])}\n\n")
            
            # Write questions
            for q in cluster_data["questions"]:
                question_id = q["question_id"]
                # Use original question text if available
                question_text = original_questions.get(question_id, q["question_text"])
                f.write(f"{question_id}. {question_text}\n")
            
            f.write("\n\n")
    
    print(f"Readable output saved to {output_file}")

def main():
    # Load clusters and original questions
    clusters = load_clusters()
    original_questions = load_original_questions()
    
    # Check if OpenAI API key is available
    use_llm = openai.api_key is not None
    if not use_llm:
        print("OpenAI API key not found in .env file. Using keyword-based labeling only.")
    else:
        print("OpenAI API key found. Using LLM for generating labels.")
    
    # Label clusters
    labeled_clusters = label_clusters(clusters, original_questions, use_llm=use_llm)
    
    # Save labeled clusters
    save_labeled_clusters(labeled_clusters)
    
    # Generate readable output
    generate_readable_output(labeled_clusters, original_questions)
    
    # Print summary
    print("\nCluster Labeling Summary:")
    for label, cluster_data in labeled_clusters.items():
        print(f"\n{label} ({len(cluster_data['questions'])} questions)")
        print(f"Keywords: {', '.join(cluster_data['keywords'])}")
        print(f"Original cluster: {cluster_data['original_name']}")

if __name__ == "__main__":
    main() 