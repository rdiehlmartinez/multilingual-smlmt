# Problyglot : A framework for Probabilistic Graphical Language Training

Problyglot is a methodology for training multi-lingual language models using a probabilistic meta-learning training approach. The main idea is to train a language multi-lingual model, by forcing the model at each training step to sample weights that will help it solve a language modeling task in a specific language. Through the training process, the goal is that the model will learn to select weights that are 'useful' for a given language and which in turn should be a good starting-off point for the model to be fine-tuned on a downstream NLU task for any language. 

The entry point to the model is run_model which expects to be passed in a config file. 

As a baseline to the probabilistic MAML approach we also implement a standard MAML approach, as well as a baseline model that simply does normal multi-lingual masked language modeling. 
