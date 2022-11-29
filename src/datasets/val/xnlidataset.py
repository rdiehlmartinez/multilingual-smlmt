__author__ = 'Richard Diehl Martinez' 
""" Dataset class for iterating over XNLI data"""

import logging
import os

from collections import defaultdict
from typing import Tuple, List

from transformers import XLMRobertaTokenizer

from .nludataset import NLUTaskDataGenerator, NLUDataset

logger = logging.getLogger(__name__)

# We always use the XLM sentencepiece tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

# imports for type hints
from configparser import ConfigParser
from typing import Dict, List, Tuple

# to stop the huggingface tokenizer from giving the sequence longer than 512 warning 
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

class XNLIDataGenerator(NLUTaskDataGenerator):

    ''' 
    A generator that yields XNLIDataset classes and contains arguments for how to setup and run 
    evaluation of the XNLI task. 
    '''

    def __init__(self, config: ConfigParser) -> None:
        """
        We assume that the data for the xnli task has been downloaded as part of the 
        XTREME cross-lingual benchmark (https://github.com/google-research/xtreme).

        Initializes a generator class that yields at each iteration a tuple of 
        Dataset objects each representing a language in the XNLI corpus which are
        the data for (respectively) finetuning and evaluating the model on.
        """
        # location of folder containing xnli data
        self.root_path = config.get("XNLI", "root_path")
        self.translated_root_path = os.path.join(self.root_path, "translate-train")

        # whether the evaluation is going to be done on dev or on test
        self.evaluation_partition = config.get("XNLI", "evaluation_partition")

        self.language_files = self._get_language_files(self.root_path)

        self._task_head_init_method = config.get("XNLI", "task_head_init_method")
        
    def _get_language_files(self, root_path: str) -> List[Dict[str, Dict[str, str]]]:
        """ 
        Helper function for setting up finetune-evaluation data pairs.

        Args:
            * root_path: path to the folder containing the xnli data

        Returns: 
            * language_files: list of dictionaries of the following format: 
                {
                    "finetune": {finetune language (str), finetune data file path (str)},
                    "evaluation": {evaluation language (str), evaluation data file path)}
                }
        """
        language_files = []
        file_paths = os.listdir(root_path)

        eng_lng_str = 'en'
        eng_file_path = os.path.join(root_path, 'train-en.tsv')

        translated_file_paths = os.listdir(self.translated_root_path)

        for file_path in file_paths: 
            file_path_split = file_path.split('-')
            file_path_partition = file_path_split[0]
            file_path_lng = file_path_split[1].split('.')[0] # removing .tsv 

            if file_path_partition != self.evaluation_partition:
                continue

            language_file_dict = dict()

            if file_path_lng != "en":
                # looking up the translated version of the current evaluation file
                # except when the eval language is already english 
                translated_file_path = list(filter(lambda x: file_path_lng in x,
                                                   translated_file_paths))[0]
                translated_full_file_path = os.path.join(self.translated_root_path,
                                                         translated_file_path)
                language_file_dict['finetune'] = {"lng": file_path_lng,
                                                  "file_path": translated_full_file_path}
            else:
                language_file_dict['finetune'] = {"lng": eng_lng_str, "file_path": eng_file_path}

            full_file_path = os.path.join(root_path, file_path)
            language_file_dict['evaluation'] = {"lng": file_path_lng, "file_path": full_file_path}

            language_files.append(language_file_dict)
            
        return language_files

    @property
    def num_classes(self) -> int:
        # contradiction, entailment, neutral 
        return 3

    @property
    def task_type(self) -> str: 
        return 'classification'

    @property
    def task_head_init_method(self) -> str:
        return self._task_head_init_method

    def __iter__(self):
        """ 
        At each iteration yields a batch of support data and an iterable evaluation dataset which
        are used for finetuning and evaluating the trained model on. 
        """

        for language_file_dict in self.language_files:
            # Since we are doing few-shot learning we want to use the translated data (except)
            # english which did not need to be translated 
            is_translated = language_file_dict['finetune']['lng'] != "en"

            finetune_dataset = XNLIDataset(
                lng=language_file_dict['finetune']['lng'],
                file_path=language_file_dict['finetune']['file_path'],
                translated=is_translated,
            )

            evaluation_dataset = XNLIDataset(
                lng=language_file_dict['evaluation']['lng'],
                file_path=language_file_dict['evaluation']['file_path'],
                translated=False,
            )

            yield (finetune_dataset, evaluation_dataset)


class XNLIDataset(NLUDataset):

    # default value for XNLI 
    MAX_SEQ_LENGTH = 128

    # xnli classes
    LABEL_MAP = {
        "contradiction": 0,
        "entailment": 1,
        "neutral": 2
    }

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
        split_line = line.split('\t')

        if self.translated:
            text_a = split_line[2]
            text_b = split_line[3]
            label = split_line[4].strip()
            if label == 'contradictory':
                label = 'contradiction'
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
            max_length=XNLIDataset.MAX_SEQ_LENGTH
        )
        input_ids = inputs['input_ids']

        return (label_id, input_ids)

def main():
    """ Basic testing of XNLI Dataset"""
    from configparser import ConfigParser

    from .nludataloader import NLUDataLoader

    config = ConfigParser()
    config.add_section('XNLI')
    config.set('XNLI', 'root_path', '../../rds-personal-3CBQLhZjXbU/data/xtreme/download/xnli')
    config.set('XNLI', 'evaluation_partition', "dev")
    config.set('XNLI', 'k', '16')

    config.add_section('LEARNER')
    config.set('LEARNER', 'method', 'maml')

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

        print(support_batch['input_ids'].shape)
        print(support_batch['label_ids'].shape)
        print(support_batch['label_ids'])


        for batch in eval_dataloader:
            print("EVAL DATASET")
            print(batch['input_ids'].shape)
            print(batch['label_ids'].shape)

            break

        exit()
        
        if finetune_dataset.language == "en":
            adaptation_batch = finetune_dataset.get_adaptation_batch()
            print(adaptation_batch)
            exit()

if __name__ == '__main__':
    main()

