"""Conservative descriptor normalization; pure code, no empirical I/O.

Merge recognizable naming variants, preserve ingredient blends and dessert
styles, and retain any descriptor with unexplained text as its own flavor.
"""
import re
import unicodedata

# Documented recognizable abbreviations in the existing aggregate flavor table.
ALIASES = {
    'st': 'strawberry', 'strb': 'strawberry',
    'bb': 'blueberry', 'rby': 'raspberry', 'bn': 'banana',
    'pln': 'plain', 'unflavored': 'plain',
    'klp': 'key lime pie', 'pomeg': 'pomegranate',
    'pomegrana': 'pomegranate',
    'mss': 'mousse',
}
# Remove only named marketing / texture variants, not ingredients or blends.
MODIFIERS = {
    'very', 'natural', 'creamy', 'thick', 'ick', 'classic', 'french',
    'red', 'royal', 'garden', 'patch', 'mist', 'bliss', 'burst', 'sensation',
    'harvest', 'harve', 'sunrise', 'sunris', 'luscious', 'cream', 'creme',
    'and', 'n', 'with',
}
COMPONENTS = {
    'plain': r'plain',
    'strawberry': r'strawberr(?:y|ies)',
    'blueberry': r'blueberr(?:y|ies)',
    'raspberry': r'raspberr(?:y|ies)',
    'blackberry': r'blackberr(?:y|ies)',
    'cranberry': r'cranberr(?:y|ies)',
    'boysenberry': r'boysenberr(?:y|ies)',
    'marionberry': r'marionberr(?:y|ies)',
    'huckleberry': r'huckleberr(?:y|ies)',
    'mixed_berry': r'mixed berr(?:y|ies)',
    'wild_berry': r'wild berr(?:y|ies)',
    'berry_unspecified': r'berr(?:y|ies)',
    'black_cherry': r'black cherry',
    'cherry': r'cherr(?:y|ies)',
    'peach': r'peach(?:es)?',
    'apricot': r'apricot', 'nectarine': r'nectarine',
    'banana': r'banana', 'mango': r'mango', 'pineapple': r'pineapple',
    'apple': r'apple', 'pear': r'pear', 'coconut': r'coconut',
    'blood_orange': r'blood orange', 'orange': r'orange',
    'key_lime': r'key lime', 'lemon': r'lemon', 'lime': r'lime',
    'tangerine': r'tangerine', 'grapefruit': r'grapefruit',
    'passion_fruit': r'passion fruit', 'kiwi': r'kiwi',
    'pomegranate': r'pomegranate', 'acai': r'acai', 'goji': r'goji',
    'rhubarb': r'rhubarb', 'prune': r'prune',
    'vanilla': r'vanilla(?: bean)?',
    'white_chocolate': r'white chocolate', 'chocolate': r'chocolat(?:e)?',
    'honey': r'honey', 'maple': r'maple', 'caramel': r'caramel',
    'coffee': r'coffee', 'cinnamon': r'cinnamon', 'mint': r'mint',
    'cheesecake': r'cheesecak(?:e)?', 'shortcake': r'shortcake',
    'pie': r'pie', 'cake': r'cake', 'mousse': r'mousse',
    'meringue': r'meringue', 'cobbler': r'cobbler',
    'granola': r'granola', 'cereal': r'cereal', 'oat': r'oats?',
    'almond': r'almonds?', 'cookie': r'cookies?',
    'smoothie': r'smoothie',
}
PATTERNS = [(name, re.compile(r'\b(?:' + expression + r')\b'))
            for name, expression in COMPONENTS.items()]


def normalize_label(label):
    text = unicodedata.normalize('NFKC', str(label)).lower().strip()
    return re.sub(r'\s+', ' ', text)


def classify(label):
    """Return (family, matched-rule); no matching ingredient text is discarded.

    Unknown abbreviations are intentionally not guessed. A conservative fallback
    retains the original normalized label, including punctuation, separately.
    """
    original = normalize_label(label)
    text = re.sub(r'[^a-z0-9]+', ' ', original).strip()
    text = ' '.join(ALIASES.get(token, token) for token in text.split())
    components = set()
    for name, pattern in PATTERNS:
        if pattern.search(text):
            components.add(name)
            text = pattern.sub(' ', text)
    remaining = [token for token in text.split() if token not in MODIFIERS]
    if components and not remaining:
        return ' + '.join(sorted(components)), 'recognized_components'
    return 'unresolved: ' + (original or 'unknown'), 'retain_exact_label'
