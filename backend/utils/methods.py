import re

def remove_punctuation(text):
    return re.sub(r'[^\w\s.!?‽？！，。]', '', text)