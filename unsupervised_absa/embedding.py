from flair.embeddings import (
    TransformerDocumentEmbeddings,
    WordEmbeddings,
    FlairEmbeddings,
    TransformerWordEmbeddings,
    StackedEmbeddings,
)
from flair.data import Sentence
from enum import Enum
import torch
import flair
from typing import List, Union
from tqdm import tqdm
from loguru import logger

POOLING = ["first", "last", "first_last", "mean"]


class ModelType(Enum):
    TRANSFORMER_DOC = TransformerDocumentEmbeddings
    TRANSFORMER_WORD = TransformerWordEmbeddings
    WORD = WordEmbeddings
    FLAIR = FlairEmbeddings


class ExtractEmbedding:
    def __init__(
        self,
        model_types: Union[List[ModelType], ModelType],
        model_names: Union[List[str], str],
        pooling_method: str = "mean",
    ) -> None:
        """

        Args:
            model_types (Union[List[ModelType], ModelType]): _description_
            model_names (Union[List[str], str]): _description_
            pooling_method (str, optional): _description_. Defaults to "mean".

        Raises:
            ValueError: _description_
        """
        assert (
            pooling_method in POOLING
        ), f"Invalid pooling method, options {POOLING.__str__}"
        # Single Model
        if isinstance(model_types, ModelType) and isinstance(model_names, str):
            self.model = self._init_model(model_types, model_names, pooling_method)
        # Stacked Model Embedding
        elif isinstance(model_types, list) and isinstance(model_names, list):
            assert len(model_types) == len(
                model_names
            ), f"`model_types` and `model_names` must be of equal length. Got lengths {len(model_types)} and {len(model_types)}"  # check length are the same
            # Stack the models
            self.models = []
            for model_type, model_name in zip(model_types, model_names):
                model = self._init_model(model_type, model_name, pooling_method)
                self.models.append(model)
            self.model = StackedEmbeddings(self.models)
        else:
            raise ValueError(
                f"Incompatible types given. Got `model_types` of type {type(model_types)} and `model_names` of type {type(model_names)}"
            )

        if torch.cuda.is_available():
            flair.device = torch.device("cuda")

    def extract(self, texts: list):
        embeddings = {}
        for text in tqdm(texts):
            sentence = Sentence(text)
            self.model.embed(sentence)
            # extract the embeddings for the text
            if isinstance(self.model, StackedEmbeddings):
                # extract embeddings for all the models given
                doc_embedding = sentence.embedding
                word_embedding = torch.mean(
                    torch.stack([word.embedding for word in sentence]), axis=0
                )
                embedding = torch.concatenate((word_embedding, doc_embedding))
            elif isinstance(self.model, TransformerDocumentEmbeddings):
                embedding = sentence.embedding
            else:
                embedding = torch.mean(
                    torch.stack([word.embedding for word in sentence]), axis=0
                )
            embeddings[text] = embedding.detach().cpu().numpy()
        return embeddings

    def _init_model(
        self, model_type: ModelType, model_name: str, pooling_method: str
    ) -> Union[
        TransformerDocumentEmbeddings,
        WordEmbeddings,
        FlairEmbeddings,
        TransformerWordEmbeddings,
    ]:
        model_type = model_type.value
        if isinstance(model_type, TransformerDocumentEmbeddings) or isinstance(
            model_type, TransformerWordEmbeddings
        ):
            model = model_type(model_name, subtoken_pooling=pooling_method)
        else:
            model = model_type(model_name)
        return model
