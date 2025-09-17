import fitz  # PyMuPDF

doc = fitz.open("Dataset.pdf")
for page in doc:
    text = page.get_text()
    print("Page length:", len(text))
