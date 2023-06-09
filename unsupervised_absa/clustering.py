from loguru import logger
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Union

import joblib
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import (
    normalized_mutual_info_score,
    silhouette_score,
    silhouette_samples
)

import numpy as np
import torch

# def load_embeddings():
#     pass


class ClusteringModel:
    """
    A wrapper class for the sklearn clustering models. With this class clustering with the library 'flair' can be done.
    """

    def __init__(self, model: Union[ClusterMixin, BaseEstimator]):
        """
        :param model: the clustering algortihm from sklearn this wrapper will use.
        :param embeddings: the flair DocumentEmbedding this wrapper uses to calculate a vector for each sentence.
        """
        self.model = model

    def fit(
        self,
        embeddings: Union[torch.tensor, np.array, list],
        cosine_distance: bool = True,
        **kwargs
    ) -> dict:
        if torch.is_tensor(embeddings):
            embeddings = embeddings.detach().numpy()
        elif isinstance(embeddings, np.ndarray):
            embeddings = embeddings
        elif isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        else:
            logger.error("Make sure to input only type numpy, tensor or list")
        if cosine_distance:
            length = np.sqrt((embeddings**2).sum(axis=1))[:, None]
            embeddings = embeddings / length

        logger.info(
            "Start clustering "
            + str(self.model)
            + " with "
            + str(len(embeddings))
            + " Datapoints."
        )
        class_labels = self.model.fit_predict(embeddings, **kwargs)
        sample_silhouette_values = silhouette_samples(embeddings, class_labels)
        result = {
            "Silhoutte Score": silhouette_score(embeddings, class_labels),
            "Sample Silhoutte": sample_silhouette_values,
            "Labels": class_labels 
        }
        return (result, embeddings)

    def predict(self, embeddings: Union[torch.tensor, np.array, list]):
        if torch.is_tensor(embeddings):
            embeddings = embeddings.detach().numpy()
        elif isinstance(embeddings, np.ndarray):
            embeddings = embeddings
        elif isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        else:
            logger.error("Make sure to input only type numpy, tensor or list")
        logger.info(
            "Start the prediction "
            + str(self.model)
            + " with "
            + str(len(embeddings))
            + " Datapoints."
        )
        predict = self.model.predict(embeddings)
        logger.info(type(predict))
        logger.info("Finished prediction and labeled all sentences.")
        return predict

    def save(self, model_file: Union[str, Path]):
        """
        Saves current model.
        :param model_file: path where to save the model.
        """
        joblib.dump(pickle.dumps(self), str(model_file))

        logger.info("Saved the model to: " + str(model_file))

    @staticmethod
    def load(model_file: Union[str, Path]):
        """
        Loads a model from a given path.
        :param model_file: path to the file where the model is saved.
        """
        logger.info("Loading model from: " + str(model_file))
        return pickle.loads(joblib.load(str(model_file)))

    def evaluate(
        self,
        embeddings: Union[torch.tensor, np.array, list],
        label: Union[torch.tensor, np.array, list],
    ):
        """
        This method calculates some evaluation metrics for the clustering.
        Also, the result of the evaluation is logged.
        :param corpus: the flair corpus this wrapper will use for evaluation.
        :param label_type: the label from the sentence will be used for the evaluation.
        """

        predict = self.predict(embeddings)
        logger.info("NMI - Score: " + str(normalized_mutual_info_score(predict, label)))
