__author__ = "Richard Diehl Martinez"
""" Dataset class for iterating over XNLI data"""

import abc
import logging
import os

from collections import defaultdict
from typing import Tuple, List

from transformers import XLMRobertaTokenizer

from .nludataset import NLUTaskDataGenerator, NLUDataset

logger = logging.getLogger(__name__)

# We always use the XLM sentencepiece tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

# imports for type hints
from configparser import ConfigParser
from typing import Dict, List, Tuple

# to stop the huggingface tokenizer from giving the sequence longer than 512 warning
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


class XNLIDataGenerator(NLUTaskDataGenerator):

    """
    A base class that yields XNLIDataset classes and contains arguments for how to setup and run
    evaluation of the XNLI task. NOTE, this class should not be directly instantiated, rather one
    of its children should be instantiated which define a specific type of evaluation setup to use
    for the XNLI task. The children of this class are as follows:

        * XNLIFewShotDataGenerator
        * XNLIStandardDataGenerator
        * XNLICrossLingualDataGenerator
    """

    def __init__(self, config: ConfigParser) -> None:
        """
        We assume that the data for the xnli task has been downloaded as part of the
        XTREME cross-lingual benchmark (https://github.com/google-research/xtreme).

        Initializes a generator class that yields at each iteration a tuple of
        Dataset objects each representing a language in the XNLI corpus which are
        the data for (respectively) finetuning and evaluating the model on.
        """

        # NOTE: self.config_name should be set in the children of this class
        assert hasattr(self, "config_name")

        # location of folder containing xnli data
        self.root_path = config.get(self.config_name, "root_path")

        # whether the evaluation is going to be done on dev or on test
        self.evaluation_partition = config.get(self.config_name, "evaluation_partition")

        self._task_head_init_method = config.get(
            self.config_name, "task_head_init_method"
        )

        # finetune training parameters
        self._batch_size = config.getint(self.config_name, "batch_size", fallback=128)
        self._max_epochs = config.getint(self.config_name, "max_epochs", fallback=5)
        self._lr = config.getfloat(self.config_name, "lr", fallback=1e-5)

        # If the evaluation languages are not specified, then we evaluate on all languages in
        # the XNLI corpus; otherwise we restrict the evaluation to the specified languages
        self.eval_languages = config.get(
            self.config_name, "eval_languages", fallback=""
        )
        if self.eval_languages:
            self.eval_languages = self.eval_languages.split(",")

    @staticmethod
    def get_cross_lingual_language_files(
        root_path: str,
        evaluation_partition: str,
        eval_languages: List[str],
    ) -> List[Dict[str, Dict[str, str]]]:
        """
        Helper function for setting up cross-lingual evaluation data pairs, where the finetune and
        evaluation languages are different (i.e. the training data has not been translated).

        Args:
            * root_path: path to the folder containing the xnli data
            * evaluation_partition: whether the evaluation is going to be done on dev or on test
            * eval_languages: list of languages to evaluate on

        Returns:
            * language_files: list of dictionaries of the following format:
                {
                    "finetune": {finetune language (str), finetune data file path (str)},
                    "evaluation": {evaluation language (str), evaluation data file path)}
                }
        """
        language_files = []
        file_paths = os.listdir(root_path)

        eng_file_path = os.path.join(root_path, "train-en.tsv")

        for file_path in file_paths:
            file_path_split = file_path.split("-")
            file_path_partition = file_path_split[0]
            file_path_lng = file_path_split[1].split(".")[0]  # removing .tsv

            if file_path_partition != evaluation_partition:
                continue

            if eval_languages and file_path_lng not in eval_languages:
                continue

            language_file_dict = dict()

            langauge_file_dict["finetune"] = {"lng": "en", "file_path": eng_file_path}
            full_file_path = os.path.join(root_path, file_path)
            language_file_dict["evaluation"] = {
                "lng": file_path_lng,
                "file_path": full_file_path,
            }

            language_files.append(language_file_dict)

        return language_files

    @staticmethod
    def get_translated_language_files(
        root_path: str,
        evaluation_partition: str,
        eval_languages: List[str],
    ) -> List[Dict[str, Dict[str, str]]]:
        """
        Helper function for setting up finetune-evaluation data pairs, where the finetune and
        evaluation languages are the same (i.e. the training data has been translated from english
        to the evaluation language)

        Args:
            * root_path: path to the folder containing the xnli data
            * evaluation_partition: whether the evaluation is going to be done on dev or on test
            * eval_languages: list of languages to evaluate on

        Returns:
            * language_files: list of dictionaries of the following format:
                {
                    "finetune": {finetune language (str), finetune data file path (str)},
                    "evaluation": {evaluation language (str), evaluation data file path)}
                }
        """
        language_files = []
        file_paths = os.listdir(root_path)

        eng_file_path = os.path.join(root_path, "train-en.tsv")

        translated_root_path = os.path.join(root_path, "translate-train")
        translated_file_paths = os.listdir(translated_root_path)

        for file_path in file_paths:
            file_path_split = file_path.split("-")
            file_path_partition = file_path_split[0]
            file_path_lng = file_path_split[1].split(".")[0]  # removing .tsv

            if file_path_partition != evaluation_partition:
                continue

            if eval_languages and file_path_lng not in eval_languages:
                continue

            language_file_dict = dict()

            if file_path_lng != "en":
                # looking up the translated version of the current evaluation file
                # except when the eval language is already english
                translated_file_path = list(
                    filter(lambda x: file_path_lng in x, translated_file_paths)
                )[0]
                translated_full_file_path = os.path.join(
                    translated_root_path, translated_file_path
                )
                language_file_dict["finetune"] = {
                    "lng": file_path_lng,
                    "file_path": translated_full_file_path,
                }
            else:
                language_file_dict["finetune"] = {
                    "lng": "en",
                    "file_path": eng_file_path,
                }

            full_file_path = os.path.join(root_path, file_path)
            language_file_dict["evaluation"] = {
                "lng": file_path_lng,
                "file_path": full_file_path,
            }

            language_files.append(language_file_dict)

        return language_files

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def max_epochs(self) -> int:
        return self._max_epochs

    @property
    def lr(self) -> float:
        return self._lr

    @property
    def num_classes(self) -> int:
        # contradiction, entailment, neutral
        return 3

    @property
    def task_type(self) -> str:
        return "classification"

    @property
    def task_head_init_method(self) -> str:
        return self._task_head_init_method

    @property
    @abc.abstractmethod
    def config_name(self) -> str:
        """
        Name of the config section that contains the parameters for this task.
        """
        raise NotImplementedError

    def __iter__(self):
        """
        At each iteration yields a batch of support data and an iterable evaluation dataset which
        are used for finetuning and evaluating the trained model on.
        """

        for language_file_dict in self.language_files:

            # If we are training on data that is not english, then that data has been translated
            is_translated = language_file_dict["finetune"]["lng"] != "en"

            finetune_dataset_params = {
                "lng": language_file_dict["finetune"]["lng"],
                "file_path": language_file_dict["finetune"]["file_path"],
                "translated": is_translated,
                "is_few_shot": self.config_name == "XNLI_FEW_SHOT",
            }

            if self.config_name == "XNLI_FEW_SHOT":
                finetune_dataset_params["k"] = self.K
                finetune_dataset_params["n"] = self.num_classes

            evaluation_dataset_params = {
                "lng": language_file_dict["evaluation"]["lng"],
                "file_path": language_file_dict["evaluation"]["file_path"],
                "translated": False,
                "is_few_shot": False,
            }

            finetune_dataset = XNLIDataset(**finetune_dataset_params)

            evaluation_dataset = XNLIDataset(**evaluation_dataset_params)

            yield (finetune_dataset, evaluation_dataset)


class XNLIStandardDataGenerator(XNLIDataGenerator):
    def __init__(self, config: ConfigParser) -> None:
        """
        An XNLI data generator that yields an XNLI dataset for finetuning and evaluation.
        """
        super().__init__(config)
        self.language_files = self.get_translated_language_files(
            self.root_path, self.evaluation_partition, self.eval_languages
        )

    @property
    def config_name(self) -> str:
        return "XNLI_STANDARD"


class XNLIFewShotDataGenerator(XNLIDataGenerator):
    def __init__(self, config: ConfigParser) -> None:
        """
        An XNLI data generator that yields a batch of n-way k-shot support data and an XNLI dataset
        for evaluation.
        """
        super().__init__(config)
        self.language_files = self.get_translated_language_files(
            self.root_path, self.evaluation_partition, self.eval_languages
        )

        self._k = config.getint(self.config_name, "k")

        assert (
            self.K * self.num_classes
        ) % self.batch_size == 0, "K * N for NLU tasks must be divisible by batch size"

    @property
    def config_name(self) -> str:
        return "XNLI_FEW_SHOT"

    @property
    def K(self) -> int:
        # number of examples per class
        return self._k


class XNLICrossLingualDataGenerator(XNLIDataGenerator):
    def __init__(self, config: ConfigParser) -> None:
        """
        AN XNLI data generator that yields an XNLI dataset for finetuning in english and an XNLI
        dataset for evaluation in a different (low resource) language.
        """
        super().__init__(config)
        self.language_files = self.get_cross_lingual_language_files(
            self.root_path,
            self.evaluation_partition,
            self.eval_languages,
        )

    @property
    def config_name(self) -> str:
        return "XNLI_CROSS_LINGUAL"


class XNLIDataset(NLUDataset):

    # default value for XNLI
    MAX_SEQ_LENGTH = 128

    # xnli classes
    LABEL_MAP = {"contradiction": 0, "entailment": 1, "neutral": 2}

    """
    Dataset for processing data for a specific language in the XNLI corpus. 
    For batching, XNLIDataset expects to use an NLUDataLoader.
    """

    def __init__(self, translated=False, **kwargs) -> None:
        """
        For a given language string and data filepath, establishes an IterableDataset that
        iterates over and processes the XNLI data for that language. The keyword arg, translated,
        indicates whether the data has been translated in which case the data preprocessing
        differs slightly.
        """

        super().__init__(**kwargs)
        self.translated = translated

    def preprocess_line(self, line: str) -> Tuple[int, List[int]]:
        """
        For a given text input line, first splits the line into the hypothesis and the premise.
        Then tokenizes and returns the two lines.

        Args:
            * line: Line of text

        Returns:
            * label_id (int): Label for the current sample
            * input_ids (list): List of tokens of combined hypothesis and premise
        """

        # splitting information from tsv
        split_line = line.split("\t")

        if self.translated:
            text_a = split_line[2]
            text_b = split_line[3]
            label = split_line[4].strip()
            if label == "contradictory":
                label = "contradiction"
        else:
            text_a = split_line[0]
            text_b = split_line[1]
            label = split_line[2].strip()

        label_id = XNLIDataset.LABEL_MAP[label]

        # tokenizing inputs
        inputs = tokenizer.encode_plus(
            text_a,
            text_b,
            add_special_tokens=True,
            max_length=XNLIDataset.MAX_SEQ_LENGTH,
        )
        input_ids = inputs["input_ids"]

        return (label_id, input_ids)


def main():
    """Basic testing of XNLI Dataset"""
    from configparser import ConfigParser

    from .nludataloader import NLUDataLoader

    config = ConfigParser()
    config.add_section("XNLI")
    config.set(
        "XNLI", "root_path", "../../rds-personal-3CBQLhZjXbU/data/xtreme/download/xnli"
    )
    config.set("XNLI", "evaluation_partition", "dev")
    config.set("XNLI", "k", "16")

    config.add_section("LEARNER")
    config.set("LEARNER", "method", "maml")

    # config.add_section('LANGUAGE_TASK')
    # config.set('LANGUAGE_TASK', 'n', '2')
    # config.set('LANGUAGE_TASK', 'k', '2')
    # config.set('LANGUAGE_TASK', 'q', '20')
    # config.set('LANGUAGE_TASK', 'sample_size', '10_000')
    # config.set('LANGUAGE_TASK', 'buffer_size', '100_000_000')
    # config.set('LANGUAGE_TASK', 'mask_sampling_method', 'proportional')
    # config.set('LANGUAGE_TASK', 'mask_sampling_prop_rate', '0.3')
    # config.set('LANGUAGE_TASK', 'max_seq_len', '128')

    dataset_generator = XNLIDatasetGenerator(config)

    for support_batch, evaluation_dataset in dataset_generator:

        eval_dataloader = NLUDataLoader(evaluation_dataset, batch_size=64)

        print(support_batch["input_ids"].shape)
        print(support_batch["label_ids"].shape)
        print(support_batch["label_ids"])

        for batch in eval_dataloader:
            print("EVAL DATASET")
            print(batch["input_ids"].shape)
            print(batch["label_ids"].shape)

            break

        exit()

        if finetune_dataset.language == "en":
            adaptation_batch = finetune_dataset.get_adaptation_batch()
            print(adaptation_batch)
            exit()


if __name__ == "__main__":
    main()
