# Question Similarity Analysis

A Python toolkit helping teachers analyze and categorize student questions using semantic similarity to identify knowledge gaps and common misconceptions.

This repository contains tools for analyzing and categorizing questions based on their semantic similarity. It uses embeddings to represent questions and clustering techniques to group similar questions together.

## Sample Data

This repository includes a sample of 100 student questions from an introductory machine learning/data mining lecture introducing linear regression. These questions (`linear_regression_questions_raw.txt`) were collected from students during and after the lecture, covering topics such as:

- Basic concepts of linear regression
- Model assumptions
- Interpretation of coefficients
- Cost function intuition
- Gradient descent
- Basic hyperparameter optimization
- Model evaluation metrics
- Common pitfalls and misconceptions

This sample dataset demonstrates how the toolkit can help instructors identify common areas of confusion and adjust their teaching accordingly.

## Files and Outputs

- `preprocess_raw_questions.py`

  - Input: `linear_regression_questions_raw.txt`
  - Output: `linear_regression_questions_processed.txt` (cleaned and normalized questions)

- `generate_embeddings.py`

  - Input: `linear_regression_questions_processed.txt`
  - Output: `question_embeddings.csv` (vector representations of each question)

- `compute_similarity.py`

  - Input: `question_embeddings.csv`
  - Output:
    - `question_similarity_matrix.csv` (pairwise similarity scores)
    - `similarity_heatmap.png` (visualization of question similarities)
    - `clusters.txt` (initial cluster assignments)

- `label_clusters.py`
  - Input: `clusters.txt`, `linear_regression_questions_processed.txt`
  - Output: `categorized_questions.txt` (questions grouped by topic with cluster labels)

## Data Files

- `linear_regression_questions_raw.txt`: Raw student questions about linear regression
- `linear_regression_questions_processed.txt`: Preprocessed questions about linear regression
- `question_embeddings.csv`: Stores the embeddings for each question
- `question_similarity_matrix.csv`: Contains the similarity scores between questions
- `categorized_questions.txt`: Questions organized by category/cluster

## Usage

0. Install required packages (via pip) and configure environment variables:

   ```
   python -m venv embeddings-venv
   source embeddings-venv/bin/activate # On Windows: embeddings-venv\Scripts\activate
   pip install -r requirements.txt
   ```

OpenAI API key (in .env file):

    ```
    OPENAI_API_KEY=<your-openai-api-key>
    ```

1. Preprocess raw questions:

   ```
   python preprocess_raw_questions.py
   ```

2. Generate embeddings for the questions:

   ```
   python generate_embeddings.py
   ```

3. Compute similarity between questions:

   ```
   python compute_similarity.py
   ```

4. Label and categorize question clusters:
   ```
   python label_clusters.py
   ```

## Example Output

After processing the sample linear regression questions, the toolkit generates clusters that might reveal patterns like:

- Questions about interpreting coefficients
- Questions about model assumptions
- Questions about R-squared and model evaluation
- Questions about outliers and their impact
- Questions about the difference between correlation and causation

This clustering helps instructors identify which topics need more clarification or additional examples in future lectures.
