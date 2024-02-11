from gensim.corpora import Dictionary
import fitz
import gensim
from pdf_handler import PDFHandler
from gensim import corpora, models
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_relevance_score(query_tokens, document_topics):
    query_vector = dictionary.doc2bow(query_tokens)
    query_vector = gensim.matutils.corpus2dense([query_vector], num_terms=len(dictionary)).T[0]

    document_topics = [np.array(doc_topics).reshape(1, -1) for doc_topics in document_topics]

    # Calculate the cosine similarity between the query and document topic vectors
    similarity_scores = [cosine_similarity(query_vector.reshape(1, -1), doc_topics) for doc_topics in document_topics]

    # Return the average similarity score
    return sum(similarity_scores) / len(similarity_scores)


handler = PDFHandler()
user_query = "Summarize the pdfs about quantum machine learning."
user_lemma = handler.extract_topics(handler.clean_text(text=user_query))

pdf_path = "/Users/kyotun/Desktop/NLP/pdfs/quantum_ml.pdf"
text = handler.read_pdf(pdf_path=pdf_path)
cleaned_text = handler.clean_text(text)
topic_of_text = handler.extract_topics(query=cleaned_text)

dictionary = corpora.Dictionary([topic_of_text])
corpus = [dictionary.doc2bow(topic_of_text)]

lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)
document_topics = [lda_model.get_document_topics(doc) for doc in corpus]

relevance_scores = [calculate_relevance_score(user_lemma, doc_topics) for doc_topics in document_topics]

#document_relevance_info = list(zip(pdf_path, relevance_scores))

#sorted_documents = sorted(document_relevance_info, key=lambda x: x[1], reverse=True)

#for document_path, relevance_score in sorted_documents:
#    print(f"Document: {document_path}, Relevance Score: {relevance_score:.2f}")