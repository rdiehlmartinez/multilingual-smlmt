__author__ = 'Richard Diehl Martinez' 
""" Base interface class for NLUDatasets and NLUDatasetGenerators"""

import abc
import logging

from collections import defaultdict

from torch.utils.data import IterableDataset

from .metadataset import IterableLanguageTaskDataset, SPECIAL_TOKEN_IDS
from .metadataloader import MetaCollator
from .nludataloader import NLUCollator

logger = logging.getLogger(__name__)

# static method for generating N-way k-shot self-supervised meta learning tasks 
generate_N_K_samples = IterableLanguageTaskDataset.generate_N_K_samples
meta_collate = MetaCollator(return_standard_labels=False)
nlu_collate = NLUCollator()

class NLUDatasetGenerator(metaclass=abc.ABCMeta):

    def __init__(self, config):
        """
        Base class for generators that yield NLUDataset classes. Requires children to be iterators.
        """
        # When using the platipus meta-learning method, we need to generate a language task for
        # model adaptation - thus we need to store the config specifying language task generation
        if config.get("LEARNER", "method") == "platipus": 
            self.language_task_kwargs = dict(config.items("LANGUAGE_TASK"))
        else: 
            self.language_task_kwargs = None

    @property
    @abc.abstractmethod
    def num_classes(self) -> int:
        """ The number of classes in the current NLU task"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def task_type(self) -> str:
        """ The type of NLU task (e.g. classifiation)"""
        raise NotImplementedError

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError()

class NLUDataset(IterableDataset, metaclass=abc.ABCMeta):

    """
    Base dataset for processing data for a specific language of an NLU task. Should be 
    used in conjugtion with the NLUDataLoader.
    """

    def __init__(self, lng, file_path, language_task_kwargs=None, K=None, **kwargs): 
        """
        For a given language string and data filepath, establishes an IterableDataset that 
        iterates over and processes the NLU data for that language. The language_task_kwargs
        keyword arg should only be set if we are using the platipus learner method.
        If so, we can use this class to generate a batch of data that is used by platipus to adapt
        and sample the weights of the model
        """

        self._lng = lng
        self.file_path = file_path

        self.language_task_kwargs = language_task_kwargs
        self.K = K 

    @property
    def language(self):
        return self._lng

    @property
    @abc.abstractmethod
    def num_classes(self):
        """ The number of classes in the current NLU task """
        raise NotImplementedError

    @abc.abstractmethod
    def preprocess_line(self, line, process_for_adaptation=False):
        """
        For a given text input line, splits, tokenizes and otherwise preprocesses the line.
        If process_for_adaptation is set, returns an iterablable of tokenized lines.

        Args: 
            * line (str): Line of text 
            * process_for_adaptation (bool): Whether to process the line for generating a batch
                of data for model adaptation (only applicable if using platipus). 
            
        Returns: 
            If process_for_adaptation: 
                * Iterable of processed and tokenized texts
            Else:
                * input_ids (list): List of input tokens
                * label_id (int): Label for the current sample
        """
        raise NotImplementedError()


    def get_adaptation_batch(self):
        """
        This method should only ever be called if using the platipus learner method. The 
        method loops over the data and effectively generates an N-way K-shot batch of masked 
        language modeling data (like what is seen during training). The goal is to generate 
        a batch of data like the model has seen during training, in order to adapt the model 
        to the language of the NLU task. Once the data has been processed, we convert the 
        data into a batch of data that can be directly fed into the model by collating the batch 
        using meta_collate.

        Returns: 
            * adaptation_batch (dict): contains the following information
                * input_ids (torch.tensor): Input tensors of shape (N*K, max_seq_len)
                * input_target_idx (torch.tensor): Tensor indicating for each sample at what index
                    we apply the final classification layer 
                * label_ids (torch.tensor): Tensor of labels corresponding to masked out subword id
                * attention_mask (torch.tensor): Tensor indicating which tokens in input_ids are
                    not pad tokens
        """

        assert(self.language_task_kwargs is not None),\
            "language_task_kwargs cannot be None when calling get_adaptation_batch"

        curr_samples_processed = 0 
        curr_samples = []
        curr_subword_to_sample = defaultdict(list)

        max_seq_len = int(self.language_task_kwargs["max_seq_len"])
        sample_size = int(self.language_task_kwargs["sample_size"])

        N = int(self.language_task_kwargs["n"])
        K = int(self.language_task_kwargs["k"])
        Q = 0 
        mask_sampling_method = self.language_task_kwargs["mask_sampling_method"]
        mask_sampling_prop_rate = float(self.language_task_kwargs["mask_sampling_prop_rate"])

        with open(self.file_path, 'r') as f:
            for line in f: 
                for token_ids in self.preprocess_line(line, process_for_adaptation=True):
                    # preprocess_line returns an iterable of tokenized text tokens

                    if len(token_ids) > max_seq_len:
                        continue
                        
                    curr_samples.append(token_ids)

                    # Within the sample keeps track of where a given token id occurs
                    sample_tok_ids_to_idx = defaultdict(list)

                    for idx, token_id in enumerate(token_ids):
                        if token_id in SPECIAL_TOKEN_IDS:
                            # don't include special tokens 
                            continue

                        sample_tok_ids_to_idx[token_id].append(idx)

                    # We loop over the tokens we've just seen in the sample and the 
                    # corresponding indices where each token occurs, and we add that 
                    # information into the curr_subword_to_sample 
                    for token_id, sample_token_idx in sample_tok_ids_to_idx.items():
                        curr_subword_to_sample[token_id].append((curr_samples_processed,
                                                                    sample_token_idx))

                    curr_samples_processed += 1 

                    if curr_samples_processed == sample_size:

                        support_set, _ = generate_N_K_samples(curr_subword_to_sample,
                                                                curr_samples, N, K, Q,
                                                                mask_sampling_method,
                                                                mask_sampling_prop_rate,
                                                                self.language)
                        processed_batch = meta_collate([("", (support_set, _))])
                        adaptation_batch = processed_batch[1]
                        return adaptation_batch
        
        # we only get here if there are not enough samples 
        logger.warning( 
            f"Dataset for XNLI (language: {self.language}) contains < {sample_size} samples"
        )
        support_set, _ = generate_N_K_samples(curr_subword_to_sample, curr_samples, N, K, Q,
                                              mask_sampling_method, mask_sampling_prop_rate,
                                              self.language)
        processed_batch = meta_collate([("", (support_set, _))])
        adaptation_batch = processed_batch[1]
        return adaptation_batch


    def generate_N_K_samples(self): 
        """ 
        Generates an N-way K-shot support set for a given NLU task that is used to finetune a
        pretrained LM model on a given NLU task.
        """

        assert(self.K is not None),\
            "K-shot needs to be specified to generate an N-way K-shot NLU support batch sample"

        support_set = defaultdict(list)

        with open(self.file_path, 'r') as f:
            for line in f: 
                label_id, input_ids = self.preprocess_line(line)

                if len(support_set[label_id]) < self.K:
                    support_set[label_id].append(input_ids)

                if len(support_set) == self.num_classes * self.K:
                    break
                
        return nlu_collate(support_set)


    def __iter__(self): 
        """ Reads over file and preprocesses each of the lines """
        with open(self.file_path, 'r') as f:
            for line in f:
                # tokenize line - recall this method needs to implemented by a subclass
                yield self.preprocess_line(line)
