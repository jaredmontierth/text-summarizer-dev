import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import os
from urllib.parse import urlparse, unquote
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from transformers import BartForConditionalGeneration, BartTokenizer


def get_article_text(url):
    response = requests.get(url)
    encoding = response.encoding if 'charset' in response.headers.get('content-type', '').lower() else 'utf-8'
    decoded_content = response.content.decode(encoding)
    soup = BeautifulSoup(decoded_content, 'html.parser')
    title = soup.find('title').text
    paragraphs = soup.find_all('p')
    article_text = '\n'.join([p.get_text() for p in paragraphs])
    return title, article_text

# old version
# def summarize_text(text, max_length=5000):
#     summarizer = pipeline("summarization")
#     summary = summarizer(text, max_length=max_length, do_sample=False, clean_up_tokenization_spaces=True)
#     return summary[0]['summary_text']



def summarize_text_pt(text, max_length=512):
    model_name = "facebook/mbart-large-50"
    tokenizer = MBart50Tokenizer.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)

    input_ids = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=1024).input_ids
    model.config.decoder_start_token_id = tokenizer.lang_code_to_id["pt_XX"]
    summary_ids = model.generate(input_ids, num_beams=4, length_penalty=2.0, max_length=max_length, min_length=50)
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary_text




def summarize_text(text, max_length=512):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    input_ids = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=1024).input_ids
    summary_ids = model.generate(input_ids, num_beams=4, length_penalty=2.0, max_length=max_length, min_length=50)
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary_text



def summarize_article(url, language="en"):
    title, article_text = get_article_text(url)
    if language == "en":
        summarized_text = summarize_text(article_text)
    elif language == "pt":
        summarized_text = summarize_text_pt(article_text)
    else:
        raise ValueError(f"Unsupported language: {language}")

    return title, summarized_text

def write_summarized_to_file(summarized_text, url):
    # Parse the URL and extract the path
    parsed_url = urlparse(url)
    path = parsed_url.path

    # Remove the leading/trailing slashes and split the path by slashes
    path_parts = path.strip('/').split('/')

    # Get the last part of the path and decode any URL-encoded characters
    last_part = unquote(path_parts[-1])

    # Create the filename with a .txt extension
    filename = f"{last_part}.txt"

    # Write the summarized text to the file
    with open(filename, "w") as file:
        file.write(summarized_text)

    print(f"Summarized text saved to {filename}")

