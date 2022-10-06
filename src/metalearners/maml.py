__author__ = 'Richard Diehl Martinez'
"""
Implements Model Agnostic Meta Learning: https://arxiv.org/abs/1703.03400
"""

import time
import higher 
import itertools
import logging

from multiprocessing.queues import Empty as EmptyQueue

import torch
import torch.distributed as dist

from .base import MetaBaseLearner
from ..taskheads import TaskHead
from ..utils import move_to_device

logger = logging.getLogger(__name__)

class MAML(MetaBaseLearner):
    def __init__(
        self,
        base_model,
        **kwargs,
    ) -> None:
        """
        MAML implements a type of MetaBaseLearner.

        Reads in a base model and sets up all of the metalearning parameters. The core idea of 
        MAML is to train a model using a two-loop approach - an outerloop that learns the learning 
        process, and an inner loop in which the model is trained on a given meta-learning task.

        Args: 
            * base_model (implementation of BaseModel): The model to be meta-learned 
                (implementation of BaseModel)
        """

        super().__init__(base_model, **kwargs)

        # Initializing params of the functional model that will be meta-learned
        self.model_params = torch.nn.ParameterList()
        for param in base_model.parameters():
            self.model_params.append(
                torch.nn.Parameter(
                    data=param.data.to(self.base_device),
                    requires_grad=param.requires_grad
                )
            )

        self.setup_optimizer()

    ###### Helper functions ######

    def meta_params_iter(self):
        """ Returns an iterator over all of the meta parameters"""
        return itertools.chain(
            self.model_params,
            self.inner_layers_lr,
            [self.classifier_lr], 
            self.retained_lm_head.values() if self.retain_lm_head else [],
        )

    def get_task_init_kwargs(
        self,
        task_init_method: str,
        n_labels: int, 
        **kwargs
    ):
        """ 
        Override base implementation of this method to replace the model with the functional 
        model and also pass in the model params when the task head is initialized using protomaml.

        Args:
            * task_init_method (str): Method for initializing the task head
            * n_labels (int): Number of labels defined by the task (i.e. classes)
        Returns:
            * init_kwargs (dict): Keyword arguments used by the initialization function 
        """

        init_kwargs = super().get_task_init_kwargs(task_init_method, n_labels, **kwargs)
        if 'protomaml' in task_init_method:
            init_kwargs['model'] = self.functional_model
            init_kwargs['params'] = self.model_params

        return init_kwargs
    
    ###### Model training methods ######

    ### Multi Processing Helper Method
    def run_inner_loop_mp(
        self,
        rank: int,
        world_size: int,
        data_queue,
        loss_queue,
        step_optimizer, 
        num_tasks_per_iteration: int,
    ):
        """
        Entry point for running inner loop using multiple processes. Sets up DDP init process
        group, wraps learner in DDP and calls forward/backward on the DDP-wrapped model.

        Args: 
            * rank (int): Rank of current GPU 
            * world_size (int): Number of GPUs should be the same as utils.num_gpus
            * data_queue (multiprocessing.Queue): Queue from which we read passed in data
            * loss_queue (multiprocessing.Queue): Queue to which we write loss values
            * step_optimizer (multiprocessing.Event): Event to signal workers to take an optimizer
                step
            * num_tasks_per_iteration (int): Number of tasks per iteration that the user specifies
                in the experiment config file
        """

        device, ddp = self.setup_DDP(rank, world_size)

        self.functionalize_model()

        while True: 
            # The main process sends signal to update optimizers

            while True: 
                # Waiting for the next batch of data 
                # NOTE: If there is no data either 1) the dataloading pipeline is taking a while 
                # or 2) the main process is waiting for all the workers to finish 
                try:
                    batch = data_queue.get(block=False)[0]
                    break
                except EmptyQueue: 
                    pass

                if step_optimizer.is_set():
                    # make sure all workers have taken an optimizer step
                    self.optimizer_step(set_zero_grad=True)
                    dist.barrier()

                    # once all workers have update params clear the flag to continue training
                    step_optimizer.clear()

                time.sleep(1) 

            task_name, support_batch, query_batch = batch

            task_loss = ddp(self, support_batch, query_batch, device)
            task_loss = task_loss/num_tasks_per_iteration
            task_loss.backward()

            loss_queue.put([task_loss.detach().item()])

    ### Main Inner Training Loop 
    def run_inner_loop(
        self,
        support_batch_list,
        query_batch,
        device=None, 
        **kwargs
    ): 
        """
        Implements the inner loop of the MAML process - clones the parameters of the model 
        and trains those params using the support_batch_list for self.num_learning_steps number 
        of steps using SGD.
        
        Args: 
            * support_batch_list: A list of task batches, each batch of task data is represented
                as a dictionary containing the following information:
                * input_ids (torch.tensor): Input tensors of shape (N*K, max_seq_len)
                * input_target_idx (torch.tensor): Tensor indicating for each sample at what index
                    we apply the final classification layer 
                * label_ids (torch.tensor): Tensor of labels corresponding to masked out subword id
                * attention_mask (torch.tensor): Tensor indicating which tokens in input_ids are
                    not pad tokens
            * query_batch: Same as the dictionary structure of the support_batch, but for the data
                of the query set 
            * device: Optional string to specify a device to override base_device

        Returns: 
            * loss (torch.Tensor): Loss value of the inner loop calculations
        """
        if device is None:
            device = self.base_device

        if not hasattr(self, "functional_model"):
            # If not using multiprocessing training, the first iteration of run_inner_loop
            # will have to functionalize the model 
            self.functionalize_model()

        self.functional_model.train()

        # Moving data to appropriate device
        support_batch_list = [move_to_device(support_batch, device) for support_batch in \
                                support_batch_list]
        query_batch = move_to_device(query_batch, device)
       
        num_inner_steps = self.num_learning_steps

        # Setting up LM head for task training (either using existing one or setting up new one)
        if self.retain_lm_head:
            lm_head = self.retained_lm_head
        else:
            init_kwargs = self.get_task_init_kwargs(
                self.lm_head_init_method,
                self.lm_head_n,
                data_batch=support_batch_list[0] if 'protomaml' in self.lm_head_init_method else None, 
                device=device
            )

            lm_head = TaskHead.initialize_task_head(
                task_type='classification',
                method=self.lm_head_init_method,
                init_kwargs=init_kwargs
            )
            
            if 'protomaml' in self.lm_head_init_method and self.use_multiple_samples:
                # If we're using protomaml, the first batch is used for sampling the task head 
                support_batch_list = support_batch_list[1:]
                num_inner_steps = self.num_learning_steps - 1 

        # NOTE: anytime we update the lm head we need to clone the params
        adapted_lm_head = {key: torch.clone(param) for key, param in lm_head.items()}

        # adapting params to the support set -> adapted params are phi
        phi = self._adapt_params(
            data_batch_list=support_batch_list, 
            params=self.model_params, 
            lm_head_weights=adapted_lm_head,
            learning_rate=self.inner_layers_lr,
            num_inner_steps=num_inner_steps,
            clone_params=True,
            optimize_classifier=True
        )

        # evaluating on the query batch using the adapted params phi  
        self.functional_model.eval()

        outputs = self.functional_model.forward(
            input_ids=query_batch['input_ids'],
            attention_mask=query_batch['attention_mask'],
            params=phi
        )

        self.functional_model.train()

        _, loss = self._compute_task_loss(
            outputs,
            query_batch,
            adapted_lm_head, 
            task_type='classification'
        )

        return loss


    ###### Model evaluation methods ######

    def run_finetuning(
        self,
        task_type,
        finetune_dataloader,
        n_labels,
        task_head_init_method,
        max_finetuning_batch_steps=-1,
        **kwargs
    ): 
        """
        Creates a copy of the trained model parameters and continues to finetune these 
        parameters on a given dataset. 
        
        Args: 
            * task_type (str): Type of task (e.g. 'classification')
            * finetune_dataloader (torch.data.Dataloader): The dataset for finetuning the model on
                is passed in as a dataloader (i.e. NLUDataloader)
            * n_labels (int): The number of labels in the given finetuning task
            * task_head_init_method (str): Method for initializing task head 
            * max_finetuning_batch_steps (int): Optional maximum number of batch steps to take 
                for model finetuning 

        Returns:
            * inference_params dict containing: 
                * finetuned_params ([nn.Parameter]): List of the finetuned model's parameters
                * task_head_weights (dict): Weights of task head (classifier head)
        """

        if not hasattr(self, "functional_model"):
            # NOTE: Edge case for if the model is only being evaluated without having 
            # been trained
            self.functionalize_model()

        self.functional_model.train()

        ### Initializing the task head used for the downstream NLU task
        task_init_data_batch = move_to_device(next(iter(finetune_dataloader)), self.base_device)
        init_kwargs = self.get_task_init_kwargs(
            task_head_init_method,
            n_labels,
            data_batch=task_init_data_batch
        )
        task_head_weights = TaskHead.initialize_task_head(
            task_type=task_type,
            method=task_head_init_method,
            init_kwargs=init_kwargs
        )

        # detaching parameters from original computation graph to create new leaf variables
        finetuned_model_params = []
        for p in self.model_params:
            detached_p = p.clone().detach()
            detached_p.requires_grad = p.requires_grad
            finetuned_model_params.append(detached_p)

        finetuned_task_head_weights = {}
        for k, p in task_head_weights.items():
            detached_p = p.detach()
            detached_p.requires_grad = True
            finetuned_task_head_weights[k] = detached_p

        finetune_params = itertools.chain(finetuned_model_params,
                                          finetuned_task_head_weights.values())
        finetune_optimizer = torch.optim.Adam(params=finetune_params)

        for batch_idx, data_batch in enumerate(finetune_dataloader):
            data_batch = move_to_device(data_batch, self.base_device)
            finetune_optimizer.zero_grad()

            # run SGD on the finetuned parameters

            outputs = self.functional_model.forward(input_ids=data_batch['input_ids'],
                                                    attention_mask=data_batch['attention_mask'],
                                                    params=finetuned_model_params)

            _, loss = self._compute_task_loss(outputs, data_batch, finetuned_task_head_weights, 
                                              task_type=task_type)

            loss.backward()
            finetune_optimizer.step()

            if max_finetuning_batch_steps > 0 and (batch_idx + 1) >= max_finetuning_batch_steps:
                break

        inference_params = {
            "finetuned_params": finetuned_model_params, 
            "task_head_weights": finetuned_task_head_weights
        }

        return inference_params


    def run_inference(self, task_type, inference_dataloader, finetuned_params, task_head_weights,
                      **kwargs):
        """ 
        This method is to be called after run_finetuning. 
        
        As the name suggests, this method runs inference on an NLU dataset for some task.

        Args: 
            * task_type (str): Type of task (e.g. 'classification')
            * inference_dataloader (torch.data.Dataloader): The dataset for inference is passed
                in as a dataloader (in most cases this will be an NLUDataloader)
            * finetuned_params ([nn.Parameter]): List of the finetuned model's parameters
            * task_head_weights (dict): Weights of task head (classifier head)

        Returns: 
            * predictions ([int]): A dictionary storing the model's predictions for each 
                datapoint passed in from the inference_dataloader as an int. 
            * loss (int): The value of the classification loss on the inference dataset.
        """
        if not hasattr(self, "functional_model"):
            # If the model is only being evaluated (and not being finetuned) it might not have
            # a functionalized version
            self.functionalize_model()
        
        predictions = []
        total_loss = 0.0
        total_samples = 0

        # Running final inference script over the evaluation data
        with torch.no_grad():
            self.functional_model.eval()

            for data_batch in inference_dataloader: 
                data_batch = move_to_device(data_batch, self.base_device)

                outputs = self.functional_model.forward(input_ids=data_batch['input_ids'],
                                                        attention_mask=data_batch['attention_mask'],
                                                        params=finetuned_params)

                logits, loss = self._compute_task_loss(outputs, data_batch, task_head_weights,
                                                       task_type=task_type)

                predictions.extend(torch.argmax(logits, dim=-1).tolist())

                batch_size = logits.size(0)
                total_loss += loss.item() * batch_size # loss is averaged across batch
                total_samples += batch_size 

            total_loss /= total_samples

            self.functional_model.train()

        return (predictions, total_loss)
