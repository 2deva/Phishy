from flask import Flask, render_template, request, jsonify
import requests
from bs4 import BeautifulSoup
import re
import joblib
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from urllib.parse import urlparse

import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Initialize the Flask application
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the pre-trained model and other components
model = joblib.load('xgboost_model.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
truncated_svd = joblib.load('truncated_svd.joblib')

# Initialize PorterStemmer and Stop Words
nltk.download('punk')
nltk.download('stopwords')
porter_stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


# Define helper functions
def get_base_domain(url):
    try:
        domain = urlparse(url).netloc
        return '.'.join(domain.split('.')[-2:])
    except Exception:
        return None


def is_suspicious_form_action(action, domain_name):
    if not action or not action.startswith('http'):
        return True
    action_domain = get_base_domain(action)
    site_domain = get_base_domain(domain_name)
    return action_domain != site_domain


def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text.lower())
    tokens = word_tokenize(text)
    stemmed_tokens = [porter_stemmer.stem(token) for token in tokens if token not in stop_words and token.isalpha()]
    return ' '.join(stemmed_tokens)


def extract_f1_features(html_content):
    cleaned_text = preprocess_text(html_content)
    tfidf_vector = tfidf_vectorizer.transform([cleaned_text])
    svd_features = truncated_svd.transform(tfidf_vector).flatten()
    f1_features = np.zeros(200)  # Initialize an array of zeros with the expected number of features
    f1_features[:len(svd_features)] = svd_features  # Assign the actual features to the array
    return f1_features


def count_total_links(soup):
    return len(soup.find_all(['a', 'img', 'script', 'link']))


def count_external_js_files(soup, total_links):
    return len(soup.find_all('script', attrs={'src': True})) / total_links if total_links > 0 else 0


def count_external_css_files(soup, total_links):
    return len(soup.find_all('link', attrs={'href': True, 'rel': 'stylesheet'})) / total_links if total_links > 0 else 0


def count_image_files(soup, total_links):
    return len(soup.find_all('img', attrs={'src': True})) / total_links if total_links > 0 else 0


def count_anchor_tags(soup, total_links):
    return len(soup.find_all('a', attrs={'href': True})) / total_links if total_links > 0 else 0


def count_empty_hyperlinks(soup, total_links):
    return len(soup.find_all('a', href=["#", "javascript:void(0);"])) / total_links if total_links > 0 else 0


def count_anchor_tags_without_href(soup, total_links):
    return len([tag for tag in soup.find_all('a') if not tag.has_attr('href')]) / total_links if total_links > 0 else 0


def count_total_hyperlinks(soup):
    return len(soup.find_all(['a', 'link', 'img', 'script']))


def count_internal_hyperlinks(soup, total_links):
    return len([tag for tag in soup.find_all(['a', 'link', 'img', 'script']) if
                tag.get('href') and not tag['href'].startswith('http')]) / total_links if total_links > 0 else 0


def count_external_hyperlinks(soup, total_links):
    return len([tag for tag in soup.find_all(['a', 'link', 'img', 'script']) if
                tag.get('href') and tag['href'].startswith('http')]) / total_links if total_links > 0 else 0


def count_external_to_internal_hyperlinks_ratio(soup, total_links):
    internal_hyperlinks = len([tag for tag in soup.find_all(['a', 'link', 'img', 'script']) if
                               tag.get('href') and not tag['href'].startswith('http')])
    external_hyperlinks = len([tag for tag in soup.find_all(['a', 'link', 'img', 'script']) if
                               tag.get('href') and tag['href'].startswith('http')])

    if internal_hyperlinks == 0:
        return 0  # Return 0 or any other default value to handle division by zero
    else:
        return external_hyperlinks / internal_hyperlinks


def is_invalid_link(link):
    return not bool(urlparse(link).netloc)


def count_error_hyperlinks(soup, total_links):
    return sum([1 for link in soup.find_all(['a', 'link', 'script', 'img']) if
                is_invalid_link(link.get('href') or link.get('src'))]) / total_links if total_links > 0 else 0


def count_total_forms(soup):
    return len(soup.find_all('form'))


def count_suspicious_form_actions(soup, domain_name):
    total_links = len(soup.find_all(['a', 'link', 'script', 'img']))
    return sum([1 for link in soup.find_all(['a', 'link', 'script', 'img']) if
                is_invalid_link(link.get('href') or link.get('src'))]) / total_links if total_links > 0 else 0


def extract_f3_to_f15_features(soup, domain_name):
    total_links = count_total_links(soup)
    f3_to_f15_features = [
        count_external_js_files(soup, total_links),
        count_external_css_files(soup, total_links),
        count_image_files(soup, total_links),
        count_anchor_tags(soup, total_links),
        count_empty_hyperlinks(soup, total_links),
        count_anchor_tags_without_href(soup, total_links),
        count_total_hyperlinks(soup),
        count_internal_hyperlinks(soup, total_links),
        count_external_hyperlinks(soup, total_links),
        count_external_to_internal_hyperlinks_ratio(soup, total_links),
        count_error_hyperlinks(soup, total_links),
        count_total_forms(soup),
        count_suspicious_form_actions(soup, domain_name)
    ]
    return f3_to_f15_features


def extract_text_features(html_content):
    cleaned_text = preprocess_text(html_content)
    tfidf_vector = tfidf_vectorizer.transform([cleaned_text])
    text_features = truncated_svd.transform(tfidf_vector).flatten()
    return text_features


def extract_features(domain_name, html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract F1 features
    f1_features = extract_f1_features(html_content)

    # Ensure F1 features are of expected length
    if len(f1_features) != 200:
        raise ValueError(f"F1 feature count mismatch. Expected: 200, Got: {len(f1_features)}")
    # Extract F3 to F15 features
    f3_to_f15_features = extract_f3_to_f15_features(soup, domain_name)
    # Extract Text Features
    text_features = extract_text_features(html_content)
    # Ensure Text features are of expected length (50 if that's what was used during training)
    if len(text_features) != 50:
        raise ValueError(f"Text feature count mismatch. Expected: 50, Got: {len(text_features)}")
    # Concatenate all features
    total_features = np.concatenate([f1_features, f3_to_f15_features, text_features])
    # Ensure total feature count matches the expected count
    if len(total_features) != 263:
        raise ValueError(f"Total feature count mismatch. Expected: 263, Got: {len(total_features)}")
    return total_features