import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset from CSV
def load_dataset(file_path):
    df = pd.read_csv(file_path, header=None, names=['intent', 'text'])
    return df

# Preprocess text data
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords and non-alphabetic characters
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    
    # Join tokens back into a string
    return ' '.join(tokens)

# Preprocess the dataset
def preprocess_dataset(df):
    texts = []
    intents = []

    for index, row in df.iterrows():
        text = row['text'].strip()
        intent = row['intent'].strip()

        # Preprocess the text
        processed_text = preprocess_text(text)
        
        # Append to lists
        texts.append(processed_text)
        intents.append(intent)
    
    return texts, intents

# Encode intents as numerical labels
def encode_intents(intents):
    label_encoder = LabelEncoder()
    encoded_intents = label_encoder.fit_transform(intents)
    return encoded_intents, label_encoder

# Main function
def main():
    # Load the dataset
    file_path = './data/raw/atis_intents_test.csv'  # Replace with your dataset path
    df = load_dataset(file_path)
    
    # Preprocess the dataset
    texts, intents = preprocess_dataset(df)
    
    # Encode intents
    encoded_intents, label_encoder = encode_intents(intents)
    
    # Create a DataFrame for easier handling
    df_processed = pd.DataFrame({'text': texts, 'intent': intents, 'encoded_intent': encoded_intents})
    
    # Display the first few rows
    print(df_processed.head())
    
    # Save the preprocessed data (optional)
    df_processed.to_csv('./data/processed/atis_test.csv', index=False)

if __name__ == '__main__':
    main()