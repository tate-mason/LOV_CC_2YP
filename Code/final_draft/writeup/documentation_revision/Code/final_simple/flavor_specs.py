"""Flavor specifications and descriptor rules. No empirical I/O."""
import re

SPECIFICATIONS = {
    'three': ['other', 'berry', 'plain'],
    'five': ['other', 'berry', 'plain', 'other_fruits', 'desserts'],
}
PLAIN = r'plain|unflavored'
# Preserve the original three-group berry rule exactly.
BERRY = r'berry|straw|blue|rasp|black|cran|cherry|wildberry'
EXPANDED_BERRY = r'berr(?:y|ies)|straw|blue|rasp|cran'
DESSERT = (r'vanilla|chocolat|\bpie\b|cheesecake|cake|cookie|cookies|brownie|'
           r'pudding|custard|caramel|toffee|butterscotch|fudge|tiramisu|'
           r'cobbler|\btart\b|dessert|sundae|ice\s*cream|creme\s*brulee|'
           r'cr[eè]me\s*br[uû]l[eé]e|dulce|marshmallow|praline|'
           r'cinnamon\s*roll|boston\s*cream|banana\s*cream')
OTHER_FRUIT = (r'cherry|cherries|peach|apricot|nectarine|mango|pineapple|banana|\bapple\b|'
               r'\bpear\b|orange|lemon|lime|citrus|grapefruit|tangerine|'
               r'mandarin|clementine|plum|prune|grape|raisin|fig|date|'
               r'coconut|passion\s*fruit|guava|papaya|kiwi|melon|'
               r'watermelon|cantaloupe|honeydew|pomegranate|lychee|'
               r'tropical|mixed\s*fruit|fruit\s*medley')


def categories(specification='three'):
    try:
        return list(SPECIFICATIONS[specification])
    except KeyError:
        raise ValueError('Unknown flavor specification: '+str(specification)) from None


def classify_descriptor(text, specification='three'):
    """Classify a descriptor; dessert beats fruit, plain retains first priority."""
    categories(specification)
    text = str(text or '').lower()
    if re.search(PLAIN, text):
        return 'plain'
    if specification == 'five' and re.search(DESSERT, text):
        return 'desserts'
    if re.search(BERRY if specification == 'three' else EXPANDED_BERRY, text):
        return 'berry'
    if specification == 'five' and re.search(OTHER_FRUIT, text):
        return 'other_fruits'
    return 'other'


def classification_expression(text, specification='three'):
    """Equivalent Polars expression for server-side reference classification."""
    import polars as pl
    categories(specification)
    expression = pl.when(text.str.contains(PLAIN)).then(pl.lit('plain'))
    if specification == 'five':
        expression = expression.when(text.str.contains(DESSERT)).then(pl.lit('desserts'))
    expression = expression.when(text.str.contains(BERRY if specification == 'three' else EXPANDED_BERRY)).then(pl.lit('berry'))
    if specification == 'five':
        expression = expression.when(text.str.contains(OTHER_FRUIT)).then(pl.lit('other_fruits'))
    return expression.otherwise(pl.lit('other'))
