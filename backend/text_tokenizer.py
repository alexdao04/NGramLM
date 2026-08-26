import spacy
from text_req_api import cleaned_text

doc = nlp(cleaned_text)
tokens = [tok.text.lower() for tok in doc if not tok.is_space and tok.is_alpha]
print(tokens[0:])
