import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def preprocess_questions(input_file, output_file):
    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Read the input file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    processed_questions = []
    
    for line in lines:
        # Extract question number and text
        match = re.match(r'(\d+)\.\s+(.*)', line.strip())
        if match:
            question_num = match.group(1)
            question_text = match.group(2)
            
            # Lowercase
            question_text = question_text.lower()
            
            # Remove punctuation (except for mathematical symbols like ², %)
            question_text = re.sub(r'[^\w\s²%]', ' ', question_text)
            
            # Handle special math symbols
            question_text = question_text.replace('²', ' squared ')
            
            # Simple tokenization by splitting on whitespace
            tokens = question_text.split()
            
            # Remove stopwords and lemmatize
            filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
            
            # Rejoin tokens
            processed_text = ' '.join(filtered_tokens)
            
            # Add to processed questions list
            processed_questions.append(f"{question_num}. {processed_text}")
    
    # Write to output file
    with open(output_file, 'w') as f:
        f.write('\n'.join(processed_questions))
    
    print(f"Preprocessing complete. Processed questions saved to {output_file}")

if __name__ == "__main__":
    input_file = "linear_regression_questions_raw.txt"
    output_file = "linear_regression_questions_processed.txt"
    preprocess_questions(input_file, output_file)
