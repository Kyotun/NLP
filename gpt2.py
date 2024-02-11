from transformers import TFAutoModelWithLMHead, AutoTokenizer
from pdf_handler import PDFHandler
import fitz
import tensorflow

handler = PDFHandler()
pdf_path = "/Users/kyotun/Desktop/NLP/pdfs/quantum_ml.pdf"
text = handler.read_pdf(pdf_path=pdf_path)

model = TFAutoModelWithLMHead.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")

# T5 uses a max_length of 512 so we cut the article to 512 tokens.
inputs = tokenizer.encode("summarize: " + text, return_tensors="tf", max_length=512)
outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)