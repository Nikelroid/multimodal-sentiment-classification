import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

download_list = [
    'punkt_tab',    # Required for word_tokenize and sent_tokenize
    'stopwords',    # Required for removing stop words
    'wordnet',      # Required for lemmatization
    'omw-1.4',      # Often required alongside wordnet in newer NLTK versions
    'averaged_perceptron_tagger_eng' # Often needed if your code does Part-of-Speech tagging
]

for package in download_list:
    nltk.download(package, quiet=True)
    
try:
    import enchant
    english_dict = enchant.Dict("en_US")
except Exception:
    english_dict = None

import emoji
from PIL import Image
import numpy as np

# Download required nltk resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    print(f"NLTK download failed: {e}")

NUM = '<NUM>'
UNK = '<UNK>'

lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = set()

def sent_preprocess(sent, lower=True, remove_punct=True, remove_stopwords=False,
                    lemmatize=False, handle_nums=False, handle_unknowns=False, remove_emojies=True, join=True):
    """
    NLP Preprocessing function mimicking the Jupyter notebook Phase 2 logic.
    """
    if lower:
        sent = sent.lower()

    if remove_punct:
        sent = sent.translate(str.maketrans('', '', string.punctuation))

    if remove_emojies:
        sent = emoji.replace_emoji(sent)

    word_tokens = word_tokenize(sent)

    if remove_stopwords and stop_words:
        word_tokens = [w for w in word_tokens if not w in stop_words]

    if lemmatize:
        word_tokens = [lemmatizer.lemmatize(w) for w in word_tokens]

    if handle_nums:
        def is_number(s):
            if s.isdigit(): return True
            if len(s) > 2 and s[:-2].isdigit() and s[-2:] in ['th', 'st', 'nd', 'rd']: return True
            return False
        word_tokens = [NUM if is_number(w) else w for w in word_tokens]

    if handle_unknowns and english_dict:
        word_tokens = [w if english_dict.check(w) else UNK for w in word_tokens]

    if join:
        return ' '.join(word_tokens)
    return word_tokens

# Lazy load RetinaFace to avoid heavy initialization if not needed
_detector = None
def get_face_detector():
    global _detector
    if _detector is None:
        try:
            from face_detection import RetinaFace
            _detector = RetinaFace(gpu_id=0) # Optionally set context
        except Exception as e:
            print(f"Warning: face_detection not installed or failed to init: {e}")
            _detector = False
    return _detector

def extract_faces(image: Image.Image, threshold=0.95):
    """
    Extract faces from a PIL Image using RetinaFace
    """
    detector = get_face_detector()
    if not detector:
        return []

    np_img = np.array(image.convert('RGB'))
    faces = []
    try:
        faces_boundaries = detector(np_img)
        for i in range(len(faces_boundaries)):
            stats, _, score = faces_boundaries[i]
            stats = stats.astype(int)
            if score > threshold:
                face_img = Image.fromarray(np_img[max(0,stats[1]):min(np_img.shape[0],stats[3]),
                                                  max(0,stats[0]):min(np_img.shape[1],stats[2])])
                faces.append(face_img)
    except Exception as e:
        print(f"Face extraction failed: {e}")
    return faces
