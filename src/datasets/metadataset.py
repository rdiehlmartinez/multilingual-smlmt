__author__ = 'Richard Diehl Martinez'
""" Utilities for preprocessing and loading dataset """

import os
import torch 
import gzip
import multiprocessing
import mmap
import random
import time
import logging
import math
import copy 
import numpy as np

from torch.utils.data import IterableDataset
from transformers import XLMRobertaTokenizer
from collections import defaultdict

logger = logging.getLogger(__name__)

# to stop the huggingface tokenizer from giving the sequence longe than 512 warning 
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# We always use the XLM sentencepiece tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
MASK_TOKEN_ID = tokenizer.mask_token_id 
SPECIAL_TOKEN_IDS = tokenizer.all_special_ids

# to encode any token id we need BYTE_ENCODING_SIZE number of bytes (hex encoding)
BYTE_ENCODING_SIZE = math.ceil(math.log(tokenizer.vocab_size + 1, 16)) 
BYTE_ENDIAN_MODE = 'big'
BYTE_END_MARKER = tokenizer.vocab_size.to_bytes(BYTE_ENCODING_SIZE, BYTE_ENDIAN_MODE)

class MetaDataset(IterableDataset):
    """
    MetaDataset that coordinates the generation of (masked language) tasks. For the 
    foreseeable future MetaDataset only supports generation of masked language tasks,
    but it can fairly trivially be adapted to also produce generation of NLU tasks. 
    """

    def __init__(self, config):
        """ 
        Initialize MetaDataset using a config file. MetaDataset is the method 
        used for training the meta learning model which is later applied to 
        downstream NLU tasks. 
        """
        logger.info(f"Initializing MetaDataset")

        languages = self._get_languages(config)
        self.datasets, self.datasets_md = self._initialize_datasets(config, languages)

        self.task_sampling_method = config.get(
            "META_DATASET",
            "task_sampling_method",
            fallback="random"
        )
        
        if self.task_sampling_method == 'proportional':
            self.task_sampling_prop_rate = config.getfloat(
                "META_DATASET",
                "task_sampling_prop_rate",
                fallback=0.5
            )
        
        super().__init__()

    @staticmethod
    def _get_languages(config): 
        """
        Helper for reading in languages from config or from a file.
        """    
        # languages_str can either be empty string, a file path or a 
        # comma-separated list of iso language codes 
        languages_str = config.get("META_DATASET", f"languages")
        if ".txt" in languages_str: 
            with open(languages_str, "r") as f: 
                languages = f.read().splitlines()
        else: 
            languages = languages_str.split(",")
    
        return languages

    def _initialize_datasets(self, config, languages):
        """ 
        Helper method for setting up datasets 
        Args: 
            * config: parsed config file passed from __init__ 
            * languages [List]: list of languages stored as iso-codes 
        Returns:
            * datasets {lng: BaseIterableDataset}: Returns a dictionary mapping a specific 
                language to the associated dataset for that language 
            * datasets_md {lng: dict()}: Returns a dictionary mapping a specific language to 
                metadata associated with the dataset for that language
        """

        def compute_dataset_size(lng_root_fp): 
            """ Calculate the size of a directory in bytes"""
            size = 0 
            for filename in os.listdir(lng_root_fp):
                filepath = os.path.join(lng_root_fp, filename)
                size += os.stat(filepath).st_size
            return size

        data_root = config.get("META_DATASET", "root_path")
        datasets = {}
        datasets_md = {}

        # passing seed to reproduce the same data by IterableLanguageTaskDataset
        seed = config.getint("EXPERIMENT", "seed", fallback=-1)

        for language in languages: 
            lng_root_fp = os.path.join(data_root, language)

            dataset_size = compute_dataset_size(lng_root_fp)

            language_task_kwargs = dict(config.items('LANGUAGE_TASK'))
            if config.getboolean("LEARNER", "use_multiple_samples", fallback=True):
                language_task_kwargs['num_task_samples'] = \
                    config.getint("LEARNER", "num_learning_steps")

            dataset = IterableLanguageTaskDataset(
                lng_root_fp,
                language,
                seed=seed,
                **language_task_kwargs
            )
            
            datasets[language] = dataset
            datasets_md[language] = {"dataset_size": dataset_size} # Can add more metadata 

        return datasets, datasets_md 

    def shutdown(self):
        """ 
        Shuts down worker nodes spawned by each of the datsets 
        """
        logger.info("Shutting down worker nodes for data processing")
        for _, dataset in self.datasets.items(): 
            dataset.shutdown()
        
        # to play nicely with wandb
        time.sleep(1)

    def __next__(self):
        """
        Called by MetaDataLoader to iterate over the dataset. First samples a language 
        (aka. a task) from which to sample a support and query set. 

        Returns: 
            * Tuple containing: 
                * sampled language (str): language of sample
                * Another tuple storing the data for the support and query sets which
                is returned from calling next on the IterableLanguageTaskDataset dataset.
        """
        # sample next task either randomly or proportional to size of dataset
        if self.task_sampling_method == 'random':
            sampled_language = random.sample(self.datasets_md.keys(), k=1)[0]
        elif self.task_sampling_method == 'proportional':
            sampling_weights = [ v['dataset_size']**self.task_sampling_prop_rate 
                                    for v in self.datasets_md.values()]
            sampling_weights = sampling_weights/np.sum(sampling_weights)
            sampled_language = random.choices(list(self.datasets_md.keys()),
                                              weights=sampling_weights, k=1)[0]
        else: 
            logger.exception(f"Invalid task sampling method: {self.task_sampling_method}")
            raise Exception(f"Invalid task sampling method: {self.task_sampling_method}")
        
        sampled_dataset = self.datasets[sampled_language]
        return (sampled_language, next(sampled_dataset))
        
    def __iter__(self):
        """ IterableDataset expects __iter__ to be overriden"""
        return self

class IterableLanguageTaskDataset(object): 
    ''' 
    Iterable dataset that reads language data from a provided directory of txt files 
    and returns at each iteration some N-way K-shot example
    '''
    def __init__(
        self,
        root_fp: str,
        lng: str,
        seed: int = -1,
        n: int = 10,
        k: int = 5,
        q: int = 10,
        num_task_samples: int = 1,
        buffer_size: int = 1e6,
        sample_size: int = 10_000,
        mask_sampling_method: str  = "proportional",
        mask_sampling_prop_rate: float = 0.3,
        max_seq_len: int = 128,
        **kwargs
    ) -> None: 
        """ 
        Initializes params and data buffers for the iterable dataset. 

        For each iteration reads in (sample_size) sentences from the dataset, and from those 
        sentences generates num_task_samples of N-way K-shot 'task' samples. This is implemented by
        first generating a single N-way (num_task_samples * K)-way task and then dividing that 
        up into num_task_samples number of samples of N-way K-shot tasks.
        Also generates a final single sample of a N-way Q-shot task that is used as the query set.

        This process happens on a worker node, which then communicates with the parent node by
        writting out the N-way K-shot data to a data buffer that is stored in shared memory. 

        Note then that when the parent calls the get_stream() method, we only need to 
        read from the data buffer which can happen quickly.
        
        Args: 
            * root_fp (str): A file path to the directory where the language data is 
                stored. This directory can have many files which are all 
                expected to be .txt.gz files, where each line is a new sample.
            * lng (str): iso code for the language corresponding to the dataset
            * seed (int): seed to use for reproducibility; a negative value will skip seed setting
            * [optional] n (int): The N in N-way K-shot classification
            * [optional] k (int): The K in N-way K-shot classification
            * [optional] q (int: The number of samples in the query set for each 'task'
                (defaults to 10) Thus for each class, we must find K+Q examples of that class.
            * [optional] num_task_samples (int): The number of samples of N-way K-shot tasks to 
                return
            * [optional] buffer_size (int): Size of the memory-mapped buffer
            * [optional] sample_size (int): Number of phrases to sample before returning a sample 
                for N-way k-shot classification 
            * [optional] mask_sampling_method (str): Either one of 'random' or 'proportional' which
                specify how to sample the N tasks
            * [optional] mask_sampling_prop_rate (float): Used if mask_sampling_method is
                'proportional', specifies the sampling proportional rate so that
                x~U(x)^{mask_sampling_prop_rate}
            * [optional] max_seq_len (int): Max length of input sequence 
        """
        super().__init__()
        self.root_fp = root_fp 
        self._lng = lng
        self.N = int(n)
        self.K = int(k) * int(num_task_samples)
        self.Q = int(q)
        self.num_task_samples = int(num_task_samples)

        buffer_size = int(buffer_size)

        # NOTE: Each sample requires roughly 1000 bytes to store (~liberal heuristic)
        if (self.N*self.K*1000 > buffer_size): 
            logger.warning(f"Small buffer size for processing lng: {lng} ({buffer_size} bytes)")

        self.sample_size = int(sample_size)
        self.mask_sampling_method = mask_sampling_method
        self.mask_sampling_prop_rate = float(mask_sampling_prop_rate)
                    
        self.max_seq_len = int(max_seq_len)
        if self.max_seq_len > tokenizer.max_len_single_sentence:
            logger.error(f"max_seq_len has to be less than {tokenizer.max_len_single_sentence}")
            logger.error(f"Overriding max_seq_len to {tokenizer.max_len_single_sentence}")
            self.max_seq_len = tokenizer.max_len_single_sentence

        # event and lock to communicate between parent and child 
        self.event = multiprocessing.Event()
        self.lock = multiprocessing.Lock()

        # Extract data out of the buffers for support and query
        self.support_data_buffer = mmap.mmap(-1, length=buffer_size)
        self.query_data_buffer = mmap.mmap(-1, length=buffer_size)
        
        self.worker = multiprocessing.Process(
            target=self.generate_buffer,
            args=(seed,)
        )
        self.event.set()
        self.worker.start()

    @property
    def language(self):
        """ Language property """
        return self._lng

    def shutdown(self): 
        """ Needs to be called in order to terminate the data generation worker """
        self.worker.terminate()
        self.worker.join()

    def split_task_samples(self, support_samples): 
        """
        Splits up support_samples into a list of length num_task_samples that each contain a 
        sample of an N-way K-shot task. 

        Args: 
            * support_samples {token_id : [K*num_task_samples samples of token_id masked out]}: 
                Mapping of N different token_ids to K samples of sentences where the token is masked out.
        Returns: 
            * task_samples ([support_samples]): A list of support_samples
        """
        task_samples = [defaultdict(list) for _ in range(self.num_task_samples)]
        for label, samples in support_samples.items():
            for task_sample_idx, task_sample in enumerate(task_samples):
                # samples is a list of length k*num_task_samples
                start_index = task_sample_idx * int(self.K/self.num_task_samples)
                end_index = (task_sample_idx+1) * int(self.K/self.num_task_samples)
                task_sample[label].extend(samples[start_index:end_index])
        return task_samples

    def __next__(self): 
        """ 
        NOTE: Called from main process
        Reads and returns the data that has been stored in the support_data_buffer and the 
        query_data_buffer by the worker node. Note that if self.num_task_samples is > 1 that 
        K will have been multiplied by this value. In this instance support_samples will be a 
        list of N-way K-shot samples.

        Returns: 
            * support_samples_list [{token_id : [K samples of token_id masked out]}]: list of length
                self.num_task_samples of N-way K-shot support samples (i.e. tasks). 
            * query_samples {token_id : [Q samples of token_id masked out]}: Mapping of 
                N different token_ids to Q samples of sentences where the token is masked out.
        """
        if self.event.is_set():
            # self.event should not be set - this can only happen on class initialization if the 
            # worker node is not fast enough to beat the main node to acquire the lock 
            time.sleep(1) 

        self.lock.acquire()

        self.support_data_buffer.seek(0)
        self.query_data_buffer.seek(0)

        support_samples = defaultdict(list)
        query_samples = defaultdict(list)

        for return_dict, data_buffer, num_samples_per_n in [(support_samples,
                                                             self.support_data_buffer,
                                                             self.K),
                                                            (query_samples,
                                                             self.query_data_buffer,
                                                             self.Q)]:
            for n in range(self.N): 

                curr_n = int.from_bytes(data_buffer.read(BYTE_ENCODING_SIZE), BYTE_ENDIAN_MODE)

                # If the bytes following the initial token_id are not the end_marker then
                # buffer state is wrong 
                assert(data_buffer.read(BYTE_ENCODING_SIZE) == BYTE_END_MARKER)

                for k in range(num_samples_per_n):
                    curr_sample = []
                    while True: 
                        curr_encoded_token = data_buffer.read(BYTE_ENCODING_SIZE)
                        if (curr_encoded_token == BYTE_END_MARKER):
                            break
                        curr_token = int.from_bytes(curr_encoded_token, BYTE_ENDIAN_MODE)
                        curr_sample.append(curr_token)
                    return_dict[curr_n].append(curr_sample)

        self.lock.release()
        self.event.set()

        support_samples_list = self.split_task_samples(support_samples)

        return (support_samples_list, query_samples)

    def __iter__(self):
        """ To comply with iterator protocol """
        return self

    # NOTE: 
    # --- The following methods should only be called by the child process ---
    
    @staticmethod
    def _tokenize_line(raw_line):
        """ Decode and tokenize a raw text string """
        decoded_line = raw_line.decode('utf-8')
        tokenized_line = tokenizer(decoded_line)
        input_ids = tokenized_line['input_ids']
        return input_ids
    
    @staticmethod
    def _process_file_paths(root_fp):
        """ Filters and shuffles the file paths stored in self.fp """
        file_paths = os.listdir(root_fp)

        # ensure all files are text files - otherwise no guarantees
        file_paths = list(filter(lambda x: ".txt" in x, file_paths))
        random.shuffle(file_paths)

        return file_paths

    @staticmethod
    def generate_N_K_samples(curr_subword_to_sample, curr_samples, N, K, Q, mask_sampling_method,
                             mask_sampling_prop_rate, language):
        """
        Given a set of samples (curr_samples) drawn from the dataset generates a sample for
        N-way K-shot classification support set + Q samples for the query set. Implemented as 
        a static method to support other classes calling this function.

        Args: 
            * curr_subword_to_sample {subword_token: [(index of occurence in curr_samples,
                                                       [indices of occurence within the sample])]:
                A dictionary mapping a current subword token to a tuple containing: 
                1. the index of the sample in curr_samples where that subword token occurs and 
                2. within the given sample all the indices of the location where that token occurs
            * curr_samples [List]: A list of self.sample_size number of samples 

            If using an instance of this class - will by defined as instance variables. 
            * N (int): Number of classes in the meta learning task 
            * K (int): Number of support samples 
            * Q (int): Number of query samples 
            * mask_sampling_method (str): Method for sampling masked words
            * mask_sampling_prop_rate (float): Sampling rate 
            * language (str): language of the N-way K-shot task

        Returns: 
            * support_set {token id: [K samples where token id occurs]}: mapping of N token ids 
                to K samples per token id occurs
            * query_set {token id: [Q samples where token id occurs]}: mapping of N token ids 
                to Q samples per token id occurs

        """
        support_set = defaultdict(list)
        query_set = defaultdict(list)

        # Filter out words that do not occur K times 
        filtered_subword_to_sample = {
            k: v for k,v in curr_subword_to_sample.items() if len(v) >= (K + Q)
        }

        # Checking whether the dataset is too small
        try: 
            assert(len(filtered_subword_to_sample) > N)
        except AssertionError:
            logger.exception(f"Cannot generate N * (K+Q) samples for dataset: {language}")
            logger.exception(f"Max possible N is: {len(filtered_subword_to_sample)}")
            raise             

        # sampling mechanism for getting the N classes
        if mask_sampling_method == 'random': 
            sampled_N = random.sample(filtered_subword_to_sample.keys(), k=N)
        elif mask_sampling_method == 'proportional':
            # samples n ~ U(n)^mask_sampling_prop_rate
            sampling_weights = np.array([len(v)**mask_sampling_prop_rate
                                            for v in filtered_subword_to_sample.values()])
            sampling_weights = sampling_weights/np.sum(sampling_weights)
            sampled_N = np.random.choice(list(filtered_subword_to_sample.keys()), N,
                                         replace=False, p=sampling_weights)
            sampled_N = sampled_N.tolist()
        else: 
            logger.exception(f"Invalid mask sampling method: {mask_sampling_method}")
            raise Exception(f"Invalid mask sampling method: {mask_sampling_method}")


        def mask_sample(k_index_information):
            """
            Given k_index_information, a tuple containing the following info,
                1. the index in curr_samples where the subword occurs
                2. within the sample, a list of the indices where the sample occurs 

            returns a sample with the correct subword masked out.
            """
           
            across_sample_index, within_sample_indices = k_index_information
            curr_sample = copy.deepcopy(curr_samples[across_sample_index])

            for within_sample_idx in within_sample_indices:
                curr_sample[within_sample_idx] = MASK_TOKEN_ID

            return curr_sample
    
        # now sample the k+q samples for each class
        for sampled_n in sampled_N: 
            # for each class i.e. n in {1,..., N} generate k sentences randomly
            # note that in a given sample there might be multiple occurences of a token
            # so we need to specify which token it is we want to mask
            subword_indices = filtered_subword_to_sample[sampled_n]
            sampled_K_plus_Q_indices = random.sample(subword_indices, k=(K + Q))

            for k_index_information in sampled_K_plus_Q_indices[:K]:
                masked_sample = mask_sample(k_index_information)
                support_set[sampled_n].append(masked_sample)

            for q_index_information in sampled_K_plus_Q_indices[K:]:
                masked_sample = mask_sample(q_index_information)
                query_set[sampled_n].append(masked_sample)

        return (support_set, query_set)

    @staticmethod
    def write_to_buffer(curr_set, curr_buffer):
        """ For the support and query set, write the data out to the respective buffers """
        for subword_id, samples in curr_set.items():
            curr_buffer.write(subword_id.to_bytes(BYTE_ENCODING_SIZE, BYTE_ENDIAN_MODE))
            curr_buffer.write(BYTE_END_MARKER)
            for sample in samples:
                for sample_tok_id in sample:
                    encoded_token_id = sample_tok_id.to_bytes(BYTE_ENCODING_SIZE, BYTE_ENDIAN_MODE)
                    curr_buffer.write(encoded_token_id)
                curr_buffer.write(BYTE_END_MARKER)
            
        curr_buffer.flush()

    def release_and_wait(self):
        """
        NOTE: This should only ever be run by a child worker.

        Helper function for releasing a lock and waiting to reacquire the lock
        to begin writing to buffer again.  
        """
        self.lock.release()
        self.event.clear()
        self.event.wait()
        self.lock.acquire() 
        self.support_data_buffer.seek(0) 
        self.query_data_buffer.seek(0)

    def generate_buffer(self, seed):
        """ 
        NOTE: This should only ever be run by a child worker. 
        This method generates a stream of data that is stored in a buffer from where it can be
        accessed by the parent process to generate support and query data. Because this method
        lives on a separate process, we need to set a seed again for this process (if we want to
        ensure reproducibility).

        Importantly, we currently continue to loop over the data by cycling over 
        the file paths indefinitely - the worker only stops when it is shut down by
        the main process.
        """

        if seed > 0: 
            random.seed(seed)
            np.random.seed(seed)

        # This lock is acquired when worker is initially launched
        self.lock.acquire()

        # keeps track of edge case where the entire dataset is smaller than self.sample_size
        is_too_small = False 
        total_samples_processed = 0 
        
        curr_samples_processed = 0 
        curr_samples = []
        curr_subword_to_sample = defaultdict(list)

        file_paths = self._process_file_paths(self.root_fp)

        while True:
            for curr_fp in file_paths:

                with gzip.open(os.path.join(self.root_fp, curr_fp)) as gzip_buffer: 
                    for curr_line in gzip_buffer:

                        if curr_samples_processed < self.sample_size: 
                            token_ids = self._tokenize_line(curr_line)

                            if len(token_ids) > self.max_seq_len:
                                # skip the current sample if it is too large for the model
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
                            total_samples_processed += 1 

                        if curr_samples_processed == self.sample_size: 
                            # done reading in all of the data 
                            support_set, query_set = self.generate_N_K_samples(
                                                                    curr_subword_to_sample,
                                                                    curr_samples,
                                                                    self.N, self.K, self.Q,
                                                                    self.mask_sampling_method,
                                                                    self.mask_sampling_prop_rate,
                                                                    self.language)

                            # writing data out to buffer 
                            try:
                                self.write_to_buffer(support_set, self.support_data_buffer)
                                self.write_to_buffer(query_set, self.query_data_buffer)
                            except ValueError as e: 
                                logger.exception(f"Buffer for dataset: {self.language} is used up")
                                raise Exception(f"Buffer for dataset: {self.language} is used up")

                            # resetting per-sample data structures 
                            curr_samples_processed = 0 
                            curr_samples = []
                            curr_subword_to_sample = defaultdict(list)
                            
                            self.release_and_wait()

            # NOTE: Just finished looping over the entire dataset 
            
            if total_samples_processed < self.sample_size: 
                # will possibly trigger after first pass through the entire dataset 

                logger.warning( 
                    f"Size of dataset for language {self.language}: {total_samples_processed} " +\
                    f"is smaller than {self.sample_size} samples")
                is_too_small = True

            if is_too_small: 
                # we have looped over entire dataset before sampling sample_size samples

                support_set, query_set = self.generate_N_K_samples(curr_subword_to_sample,
                                                                   curr_samples,
                                                                   self.N, self.K, self.Q,
                                                                   self.mask_sampling_method,
                                                                   self.mask_sampling_prop_rate,
                                                                   self.language)

                # writing data out to buffer 
                try:
                    self.write_to_buffer(support_set, self.support_data_buffer)
                    self.write_to_buffer(query_set, self.query_data_buffer)
                except ValueError as e: 
                    logger.exception(f"Buffer for dataset: {self.language} ran out of space")
                    raise Exception(f"Buffer for dataset: {self.language} ran out of space")

                # resetting per-sample data structures 
                curr_samples_processed = 0 
                curr_samples = []
                curr_subword_to_sample = defaultdict(list)
                
                self.release_and_wait()