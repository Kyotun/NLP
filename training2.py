import fitz  # PyMuPDF
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk import download
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from operator import itemgetter

# Download NLTK resources if not already downloaded
download('punkt')

def read_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()
    return text

def preprocess_text(text):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned_sentences = [sentence.lower() for sentence in sentences if sentence.lower() not in stop_words]
    return cleaned_sentences

def sentence_similarity(sent1, sent2):
    vector1 = [word.lower() for word in sent1.split()]
    vector2 = [word.lower() for word in sent2.split()]
    
    all_words = list(set(vector1 + vector2))
    
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    for word in vector1:
        if word in vector1:
            vector1[all_words.index(word)] += 1
    
    for word in vector2:
        if word in vector2:
            vector2[all_words.index(word)] += 1
    
    return cosine_similarity([vector1, vector2])[0, 1]

def build_similarity_matrix(sentences):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j])
    
    return similarity_matrix

def generate_summary(text, num_sentences=5):
    sentences = preprocess_text(text)
    
    sentence_similarity_matrix = build_similarity_matrix(sentences)
    
    scores = np.zeros(len(sentences))
    for i in range(len(sentences)):
        scores[i] = sum(sentence_similarity_matrix[i])
    
    ranked_sentences = [(score, i) for i, score in enumerate(scores)]
    ranked_sentences = sorted(ranked_sentences, key=itemgetter(0), reverse=True)
    
    selected_sentences = sorted([sentence for _, sentence_index in ranked_sentences[:num_sentences]])
    
    summary = ' '.join(selected_sentences)
    return summary

if __name__ == "__main__":
    pdf_path = "/Users/emirpisirici/Desktop/NLP/quantum_ml.pdf"  # Replace with your PDF file path
    pdf_text = read_pdf(pdf_path)
    
    summary = generate_summary(pdf_text)
    
    # print("Original Text:\n", pdf_text)
    print("\nSummary:\n", summary)