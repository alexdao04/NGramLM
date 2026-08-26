from collections import Counter, defaultdict
import random
import re
import requests
import spacy

nlp = spacy.blank("en")
nlp.max_length = 5_000_000

url = "https://en.wikipedia.org/w/api.php"
params = {
    "action": "query", # action to be done
    "format": "json", # desired return format
    "titles": "Google", # article to query. ONLY modify the value for "titles"
    "prop": "extracts", # specify for summary of article
    "explaintext": True # this makes sure that we're getting text not html slop
}
headers = {"User-Agent": "LING-144 classroom notebook (educational use)"}
response = requests.get(url, params = params, headers = headers, timeout = 30)
response.raise_for_status()

response_text = response.json() # python dict format
pages = response_text["query"]["pages"]
page_ID = list(pages.keys())[0] # fetch page ID
cleaned_text = pages[page_ID]["extract"]

print(cleaned_text[0:]) # this should return cleaned text from wikipedia
