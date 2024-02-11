from pdf_handler import PDFHandler

handler = PDFHandler()

user_query = "Summarize all the documents about gravitational anomalies in black holes horizons."
topics = handler.extract_topics(query=user_query)
print("Extracted topics:", topics)