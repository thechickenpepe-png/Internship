import fitz
import sys
sys.stdout.reconfigure(encoding='utf-8')

doc = fitz.open("Dataset.pdf")
for i, page in enumerate(doc):
    text = page.get_text().strip()
    print(f"Page {i+1} length: {len(text)}")
    print(f"Page {i+1} preview: {text[:100]}")
