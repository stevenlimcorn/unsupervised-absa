from transformers import pipeline
from typing import Union, Optional
import datasets
from datasets import Dataset
import pandas as pd
import torch
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
from transformers import ZeroShotClassificationPipeline
from loguru import logger


########################################################
##CUSTOM CLASS FOR ZERO SHOT CLASSIFICATION IN BATCHES##
########################################################
class CustomZeroShotClassificationArgumentHandler:
    """
    Handles arguments for zero-shot for text classification by turning each possible label into an NLI
    premise/hypothesis pair.
    """

    def _parse_labels(self, labels):
        if isinstance(labels, str):
            labels = [label.strip() for label in labels.split(",") if label.strip()]
        return labels

    def __call__(self, sequences, label_count, labels, hypothesis_template):
        if len(labels) == 0 or len(sequences) == 0:
            raise ValueError(
                "You must include at least one label and at least one sequence."
            )
        if hypothesis_template.format(labels[0]) == hypothesis_template:
            raise ValueError(
                (
                    'The provided hypothesis_template "{}" was not able to be formatted with the target labels. '
                    "Make sure the passed template includes formatting syntax such as {{}} where the label should go."
                ).format(hypothesis_template)
            )

        if isinstance(sequences, str):
            sequences = [sequences]

        if isinstance(labels[0], list):
            batch = True
        else:
            batch = False

        count = label_count
        sequence_pairs = []
        for idx, sequence in enumerate(sequences):
            if batch:
                sequence_pairs.extend(
                    [
                        [sequence, hypothesis_template.format(label)]
                        for label in labels[count]
                    ]
                )
                count += 1
            else:
                sequence_pairs.extend(
                    [[sequence, hypothesis_template.format(label)] for label in labels]
                )
        return sequence_pairs, sequences, count


class MyPipeline(ZeroShotClassificationPipeline):
    def __init__(
        self, args_parser=CustomZeroShotClassificationArgumentHandler(), *args, **kwargs
    ):
        self.count = 0
        super().__init__(args_parser, *args, **kwargs)

    def preprocess(
        self, inputs, candidate_labels=None, hypothesis_template="This example is {}."
    ):
        sequence_pairs, sequences, new_count = self._args_parser(
            inputs, self.count, candidate_labels, hypothesis_template
        )
        self.count = new_count
        for i, (candidate_label, sequence_pair) in enumerate(
            zip(candidate_labels, sequence_pairs)
        ):
            if isinstance(candidate_label, list):
                model_input = self._parse_and_tokenize([sequence_pair])
                yield {
                    "candidate_label": sequence_pair[1],
                    "sequence": sequences[0],
                    "is_last": i == len(candidate_label) - 1,
                    **model_input,
                }
            else:
                model_input = self._parse_and_tokenize([sequence_pair])
                yield {
                    "candidate_label": candidate_label,
                    "sequence": sequences[0],
                    "is_last": i == len(candidate_labels) - 1,
                    **model_input,
                }

    def reset_count(self):
        self.count = 0


# Extract Polarity Method
class MnliPipeline:
    def __init__(self, model_name: str) -> None:
        self.pipe = pipeline(model=model_name, pipeline_class=MyPipeline)
        self.model_name = model_name

    def extract_polarity(
        self,
        dataset: Union[datasets.arrow_dataset.Dataset, pd.DataFrame],
        text_variable_name: str,
        label_variable_name: str,
        device: Optional[Union[int, str, torch.device]] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        batch_size: int = 16,
    ) -> datasets.arrow_dataset.Dataset:
        """
        Extract polarity of each text and aspect term or category in each sentence.
        Dataset format needs to follow exactly as such
        | text      | aspectTerms or aspectCategory |
        | some text | Term                          |

        Args:
            dataset (Union[datasets.arrow_dataset.Dataset, pd.DataFrame]): dataset that follows above sample
            text_variable_name (str): column name for text
            label_variable_name (str): column name for category label
            device (Union[int, str, torch.device]): device type (cuda, mps or cpu)
            batch_size (str, optional): batch size Defaults to 16.
        """
        # check dataset being received
        assert (
            type(dataset) == pd.DataFrame
            or type(dataset) == datasets.arrow_dataset.Dataset
        ), f"Invalid data type {type(dataset)}"
        if type(dataset) == datasets.arrow_dataset.Dataset:
            df = dataset.to_pandas()
        else:
            df = dataset

        # preprocessing dataset
        logger.info(f"Preprocessing dataset with length: {len(dataset)}")
        term_dataset = self._preprocess_terms(df, label_variable_name)

        # extracting polarity
        logger.info(f"Extracting polarity with model: {self.model_name}")
        outputs = []
        try:
            for out in tqdm(
                self.pipe(
                    KeyDataset(term_dataset, text_variable_name),
                    batch_size=batch_size,
                    candidate_labels=KeyDataset(term_dataset, "aspectLabel"),
                    device=device,
                ),
                total=len(term_dataset),
            ):
                outputs.append(out)
            self.pipe.reset_count()
        except Exception as e:
            self.pipe.reset_count()
            raise e
        except KeyboardInterrupt:
            self.pipe.reset_count()
            raise KeyboardInterrupt

        # postprocessing outputs
        logger.info("Postprocessing outputs")
        return Dataset.from_pandas(self._postprocess_terms(outputs, text_variable_name))

    def _postprocess_terms(self, outputs: list, text_variable_name: str):
        array_of_dicts = []
        for output in outputs:
            array_of_dicts.append(self._get_polarity_term(output, text_variable_name))
        df = pd.DataFrame.from_dict(array_of_dicts)
        return df

    def _get_polarity_term(self, output: dict, text_variable_name: str):
        processed = {}
        term = None
        for label, score in zip(output["labels"], output["scores"]):
            # sample: This example is negative sentiment towards staff.
            polarity, term = (
                label.replace("This example is ", "")
                .replace(".", "")
                .split(" sentiment towards")
            )
            term = term
            processed[polarity] = score
        # argmax of previously extracted polarity
        processed["polarity"] = max(processed, key=processed.get)
        processed["term"] = term
        processed[text_variable_name] = output["sequence"]
        return processed

    def _preprocess_terms(
        self,
        dataset: pd.DataFrame,
        label_variable_name: str,
    ):
        new_df = dataset.copy()
        new_df["aspectLabel"] = new_df[label_variable_name].apply(
            lambda x: [
                f"positive sentiment towards {x}",
                f"negative sentiment towards {x}",
                f"neutral sentiment towards {x}",
            ]
        )
        return Dataset.from_pandas(new_df)
