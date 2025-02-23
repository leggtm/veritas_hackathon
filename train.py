# reads the pdf file and turns it into a readable text

import pdfplumber

# Open and extract text from the PDF
pdf_path = "policy.pdf"
policy_text = ""
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        policy_text += page.extract_text() + "\n"

# Install Sumy first:
# pip install sumy

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

def simple_summarizer(text, sentence_count=3):
    """
    Summarize the input text using Sumy's LSA summarizer.

    Parameters:
    - text (str): The text to be summarized.
    - sentence_count (int): Number of sentences to include in the summary.

    Returns:
    - str: The summarized text.
    """
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentence_count)
    summary_text = ' '.join(str(sentence) for sentence in summary)
    return summary_text

# Example usage:
policy_summary = simple_summarizer(policy_text, sentence_count=3)
print("Policy Summary:")
print(policy_summary)
