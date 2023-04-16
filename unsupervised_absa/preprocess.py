from typing import Union
import numpy as np
import pandas as pd
from loguru import logger
import contractions
import re
from typing import Union
from tqdm import tqdm
from unidecode import unidecode
import unicodedata
import string
import matplotlib.pyplot as plt
from collections import Counter
from wordsegment import load, segment


# Main Methods
def simple_preprocessing(
    data: Union[np.array, pd.Series, list]
) -> Union[np.array, pd.Series, list]:
    """Preprocess text dataset, getting np.array or pd.Series data type

    Args:
        data (Union[np.array, pd.Series]): _description_

    Returns:
        (Union[np.array, pd.Series]): preprocessed text data
    """
    functions = [
        encode_decode,
        convert_unicode,
        remove_url,
        remove_control_characters,
        remove_tags,
        remove_emoji,
        convert_contractions,
        remove_numbers,
        remove_punctuation,
        remove_multiple_spaces,
        strip_spaces,
    ]
    load()
    if isinstance(data, np.ndarray):
        output = data.astype(str)
        for fn in (pbar := tqdm(functions)):
            pbar.set_description(f"Processing {fn.__name__}")
            output = np.vectorize(fn)(output)
    elif isinstance(data, list):
        output = data.copy()
        for fn in (pbar := tqdm(functions)):
            pbar.set_description(f"Processing {fn.__name__}")
            output = list(map(fn, output))
    elif isinstance(data, pd.Series):
        tqdm.pandas()
        output = data.astype(str)
        for fn in (pbar := tqdm(functions)):
            pbar.set_description(f"Processing {fn.__name__}")
            output = output.apply(fn)
    else:
        logger.error("Current data types supported only series and numpy array")
        raise AssertionError()
    return output


def plot_top_k_words(text: Union[np.array, pd.Series, list], k: int):
    split_text = " ".join(text).lower().split()
    counter = Counter(split_text)
    y = [value for _, value in counter.most_common(k)]
    x = [key for key, _ in counter.most_common(k)]

    f, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.bar(x, y, width=0.8, align="center")
    ax.set_title("Top k Most Frequent Words")
    ax.ylabel = "Frequency"
    ax.tick_params(axis="x", labelrotation=90)
    for i, (key, value) in enumerate(zip(x, y)):
        ax.text(
            i, value, f" {value} ", rotation=90, ha="center", va="top", color="white"
        )
    ax.xlabel = "Words"
    f.tight_layout()
    return ax


# Helper Methods
def encode_decode(text: str) -> str:
    bytes_encoded = text.encode(encoding="utf-8")
    str_decoded = bytes_encoded.decode()
    return str_decoded


def remove_url(text: str) -> str:
    """Remove url from string

    Args:
        text (str): text to remove url

    Returns:
        str: text with url removed
    """
    text = re.sub(r"http\S+", "", text, flags=re.MULTILINE)
    return re.sub(r"www\S+", "", text, flags=re.MULTILINE)


def remove_tags(text: str) -> str:
    """Remove hashtags and @ tags from string

    Args:
        text (str): text to remove hashtags and @ tags

    Returns:
        str: text with hashtags and @ tags removed
    """
    hashtags = re.findall("#[A-Za-z0-9_]+", text)
    hash_words = [" ".join(segment(word.replace("#", ""))) for word in hashtags]
    new_text = text
    for old_word, new_word in zip(hashtags, hash_words):
        new_text = new_text.replace(old_word, new_word)
    removed_tags = re.sub("@[A-Za-z0-9_]+", "", new_text)
    return removed_tags


def remove_control_characters(text: str) -> str:
    """Remove control characters from string

    Args:
        text (str): text to remove control characters

    Returns:
        str: text with control characters removed
    """
    return re.sub(r"[\n\r\t]", " ", text)


# Reference https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
def remove_emoji(text: str) -> str:
    """Remove emoji from string

    Args:
        text (str): text to remove emoji

    Returns:
        str: text with emoji removed
    """
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002500-\U00002BEF"  # chinese char
        "\U00002702-\U000027B0"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)


def convert_contractions(text: str) -> str:
    """Convert contractions in the text, e.g (doesn't -> does not), to ensure easy preprocessing

    Args:
        text (str): text that needs to be convert

    Returns:
        str: text with contractions converted
    """
    remove_multiple_spaces = lambda x: re.sub(" +", " ", x)
    removed_spaces = remove_multiple_spaces(text)
    expanded_words = []
    for word in removed_spaces.split():
        expanded_words.append(contractions.fix(word))
    return " ".join(expanded_words)


def remove_numbers(text: str) -> str:
    """Remove numbers in the text

    Args:
        text (str): text with numbers to be removed

    Returns:
        str: text with numbers removed
    """
    return re.sub(r"[0-9]", "", text)


def convert_unicode(text: str) -> str:
    """Convert unicodes

    Args:
        text (str): Convert unicodes in text such as fancy quotes to regular quotes

    Returns:
        str: converted text
    """
    output = unicodedata.normalize("NFKD", text)
    return unidecode(output)


def remove_punctuation(text: str) -> str:
    """remove punctuations

    Args:
        text (str): remove punctuations from text

    Returns:
        str: text with punctuations removed
    """
    PUNCTUATION = string.punctuation
    return text.translate(str.maketrans(PUNCTUATION, " " * len(PUNCTUATION)))


def strip_spaces(text: str) -> str:
    return text.strip()


def remove_multiple_spaces(text: str) -> str:
    return re.sub(" +", " ", text)


# lemmatization
