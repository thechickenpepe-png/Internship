from transformers import pipeline
import sys
sys.stdout.reconfigure(encoding='utf-8')

print("⏳ Loading summarizer...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
print("✅ Summarizer loaded")

summary = summarizer("An airplane engine generates thrust by expelling air.", min_length=5, max_length=50)
print("Summary:", summary[0]["summary_text"])
