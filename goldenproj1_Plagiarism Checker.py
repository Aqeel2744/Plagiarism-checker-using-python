import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import texts 

# Load spaCy English model
nlp = spacy.load('en_core_web_sm')

# Preprocessing function using spaCy
def preprocess(text):
    text = text.lower()
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

# Function to check plagiarism
def check_plagiarism(text1, text2):
    # Preprocess both texts
    text1_processed = preprocess(text1)
    text2_processed = preprocess(text2)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1_processed, text2_processed])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    return similarity[0][0]

# getting texts from texts.py
text1 = texts.text1  
text2 = texts.text2  

#similarity checking code is here
similarity_score = check_plagiarism(text1, text2)
print(f"Similarity: {similarity_score * 100:.2f}%")

if similarity_score > 0.7:
    print("Warning: Potential plagiarism detected!")
else:
    print("No plagiarism detected.")

