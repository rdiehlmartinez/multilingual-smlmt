# Multilingual SMLMT


This library provides functionality for training a multi-lingual language model via meta learning. 
The tasks that are used to meta learn the model are variants of the Subset Masked Language Modeling
Tasks (SMLMT) as defined initially [Bansal et al.](https://arxiv.org/pdf/2009.08445.pdf). 

The contributions of this library are in the form of 1) a dataset class that given text files 
containing unlabeled text samples in different languages can generate multi-lingual SMLMT tasks,
and 2) a framework for training a language model thorugh a meta-learning mutli-level optimization 
approach on the generated SMLMT tasks, 3) a pipeline for evaluating the performance of the model 
on a set of NLU tasks. 

To run training and or evaluation of a language model via smlmt training, you must specify the 
desired configurations in a config file stored under the configs directory. 