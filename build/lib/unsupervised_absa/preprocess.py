from typing import Union
import numpy as np
import pandas as pd
from loguru import logger
import contractions
import re
from typing import Union
from tqdm.auto import tqdm
from unidecode import unidecode
import unicodedata


# Main Methods
def simple_preprocessing(
    data: Union[np.array, pd.Series]
) -> Union[np.array, pd.Series]:
    """Preprocess text dataset, getting np.array or pd.Series data type

    Args:
        data (Union[np.array, pd.Series]): _description_

    Returns:
        (Union[np.array, pd.Series]): preprocessed text data
    """
    if isinstance(data, np.ndarray):
        output = data.astype(str)
        output = np.vectorize(convert_unicode)(data)
    elif isinstance(data, pd.Series):
        pass
    else:
        logger.error("Current data types supported only series and numpy array")
        return
    return "hallo"


# Helper Methods
def remove_url(text: str) -> str:
    """Remove url from string

    Args:
        text (str): text to remove url

    Returns:
        str: text with url removed
    """
    text = re.sub(r"http\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^\s\d]+\.[^\s\d]+", "", text, flags=re.MULTILINE)
    return re.sub(r"www\S+", "", text, flags=re.MULTILINE)


def remove_hashtags(text: str) -> str:
    """Remove hashtags from string

    Args:
        text (str): text to remove hashtags

    Returns:
        str: text with hashtags removed
    """
    hashtags = re.sub("#[A-Za-z0-9_]+", "", text)
    return hashtags


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


# lemmatization
