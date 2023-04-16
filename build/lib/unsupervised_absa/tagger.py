import torch
import flair
from tqdm import tqdm
from flair.nn import Classifier
from flair.data import Sentence
from typing import Iterable, Union
from loguru import logger


class Tagger:
    def __init__(
        self,
        tagger_name: str,
        device: Union[str, torch.device] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        self.tagger_name = tagger_name
        self.tagger = Classifier.load(tagger_name)
        if isinstance(device, torch.device):
            flair.device = device
        elif isinstance(device, str):
            # check what device
            if device == "cuda" and torch.cuda.is_available():
                flair.device = torch.device(device)
            elif device == "mps" and torch.backends.mps.is_available():
                flair.device = torch.device(device)
            else:
                flair.device = torch.device("cpu")
        else:
            flair.device = torch.device("cpu")
        logger.info(f"Tagger model instantiated with device: {flair.device}")

    def tagging(self, texts: Iterable, filter_tags: list = None) -> list[dict]:
        if filter_tags != None:
            self._check_valid_tags(filter_tags=filter_tags)

        logger.info(f"Extracting {self.tagger_name} tags")
        tags = []
        for text in tqdm(texts):
            text = Sentence(text)
            text.to(device=flair.device)
            self.tagger.predict(text)
            pos_dict = []
            word = []
            start, end, tag = None, None, None
            for label in text.get_labels():
                if filter_tags == None:
                    pos_dict.append(
                        {
                            "start": label.data_point.start_position,
                            "end": label.data_point.end_position,
                            "tag": label.value,
                            "word": label.data_point.text,
                        }
                    )
                elif label.value in filter_tags:
                    if start == None:
                        start = label.data_point.start_position
                    end = label.data_point.end_position
                    tag = label.value
                    word.append(label.data_point.text)
                elif len(word) != 0:
                    pos_dict.append(
                        {
                            "start": start,
                            "end": end,
                            "tag": tag,
                            "word": " ".join(word),
                        }
                    )
                    start, end, tag = None, None, None
                    word = []
            if len(word) != 0:
                pos_dict.append(
                    {
                        "start": start,
                        "end": end,
                        "tag": tag,
                        "word": " ".join(word),
                    }
                )
                start, end, tag = None, None, None
            tags.append(pos_dict)
        return tags

    def _check_valid_tags(self, filter_tags: list) -> None:
        valid_tags = self.tagger.label_dictionary.get_items()
        invalid_tag = set(filter_tags).difference(valid_tags)
        assert len(invalid_tag) == 0, f"Invalid tags {invalid_tag}"
