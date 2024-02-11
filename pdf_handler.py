import os
import fitz
import spacy
import nltk
import re
import PyPDF2
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag



class PDFHandler:
    def __init__(self, folder_path: str = None, pdf_path: str = None) -> None:
        self.folder_path = folder_path
        self.current_pdf_path = pdf_path

    def read_pdf(self, pdf_path: str) -> str:
        """Reads all the text of the given documents with 'file_path'

        Args:
            pdf_path (_str_): File path of the document.

        Returns:
            _str_: Text of given file_path document.
        """
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text += page.get_text()
        doc.close()
        return text

    def get_pdfs(self, folder_path: str) -> list[str]:
        """Reads all the pdfs in the given folder path. Return the texts of them.

        Args:
            folder_path (str): Folder that contains pdfs.

        Returns:
            _list_: Contains true path of each pdf files in given folder path.
        """
        file_names = os.listdir(folder_path)
        pdf_file_paths = [os.path.join(folder_path, file_name) for file_name in file_names if
                          file_name.lower().endswith('.pdf')]
        return pdf_file_paths

    def lemmatize_text(self, text: str) -> list[str]:
        tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
        return lemmatized_tokens

    def extract_topics(self, query: str) -> list[str]:
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        lemmatized_words = self.lemmatize_text(text=query)
        tagged_words = pos_tag(lemmatized_words)
        topics = [word for word, pos in tagged_words if pos in ['NN', 'NNP']]
        return topics

    def clean_text(self, text: str) -> str:
        nltk.download('words')
        english_words = set(words.words())
        latin_text = ""
        latin_text = re.sub(r'[^a-zA-Z\s]', '', text)
        latin_text = self.lemmatize_text(latin_text)
        filtered_words = [word for word in latin_text if word in english_words and len(word) > 1]
        return ' '.join(filtered_words)
