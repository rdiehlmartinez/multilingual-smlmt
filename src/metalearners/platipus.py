__author__ = 'Richard Diehl Martinez'
"""
Implementation of the platipus model, proposed by Finn et el. https://arxiv.org/pdf/1806.02817.pdf
Adapted from: https://github.com/cnguyen10/few_shot_meta_learning
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
from ..utils.modeling import kl_divergence_gaussians
from ..utils import move_to_device

logger = logging.getLogger(__name__)

class Platipus(MetaBaseLearner):
    def __init__(self, base_model,  optimizer_type='adam',
                                    meta_lr=1e-2,
                                    gamma_p=1e-2,
                                    gamma_q=1e-2,
                                    inner_lr=1e-2,
                                    classifier_lr=1e-1,
                                    kl_weight=0.5,
                                    num_conditioning_steps=5,
                                    num_learning_steps=100,
                                    num_model_samples=5,
                                    use_first_order=False,
                                    language_embedding_method='val_grad',
                                    *args,
                                    **kwargs):
        """
        Platipus implements a type of BaseLearner.

        Reads in a base model and sets up all of the metalearners of the model that are going to 
        be learned, using the higher library. Platipus requires tracking meta parameters that
        represent the mean and standard deviations of the weights that are to-be meta-learned.
        These meta parameters are stored  as internal attributes of platipus, and are learned via
        an optimizer of type (optimizer_type). The main logic of platipus is in the
        run_inner_loop() and finetuning/evaluation methods which implement the training 
        and inference steps of platipus. 

        Args: 
            * base_model (implementation of BaseModel)
            * optimizer_type (str) : The type of optimizer (e.g. 'adam') 
            * meta_lr (float): Learning rate for the outer loop meta learning step
            * The following four keyword args define inner-loop hyper-parameters that can be 
                meta-learned. The values passed in represent the initial values:
                * gamma_p (float): std. dev of the gaussian dist. from which we sample weights 
                    conditioned on the support set 
                * gamma_q (float): std. dev of the gaussian dist. from which we sample weights 
                    conditioned on the query set 
                * inner_lr (float): inner-loop learning rate of the base_model 
                * classifier_lr (float): inner-loop learning rate of the classifier head
            * kl_weight (float): Trade-off hyper-parameter between the CE term and the KL term of
                the ELBO objective 
            * num_conditioning_steps (int): Number of gradients steps in the inner loop used to 
                condition the weights of the model on a given task
            * num_learning_steps (int): Number of gradients steps in the inner loop used to 
                learn the meta-learning task 
            * num_model_samples (int): Amount of times to sample weights from the model - the 
                predictions from each of these models are then combined
            * use_first_order (bool): Whether a first order approximation of higher-order gradients
                should be used (defaults to False)
            * language_embedding_method (str): How to represent the embedding that is used to
                condition the language-dependent probability distribution from which we sample
                weights for a given language (defaults to 'val_grad'). 'Val_grad' is the implicit 
                method used in the platipus paper. 
        """

        # TODO: UPDATE +ME
        raise NotImplementedError()

        super().__init__(base_model, inner_lr, classifier_lr, *args, **kwargs)
        
        # establishing meta parameters to-be learned
        self.mu_theta = torch.nn.ParameterList()
        self.log_sigma_theta = torch.nn.ParameterList()
        self.log_v_q = torch.nn.ParameterList()

        for param in base_model.parameters():
            # mean parameters initialized to weights of base model
            self.mu_theta.append(torch.nn.Parameter(data=param.data.to(self.base_device),
                                                    requires_grad=param.requires_grad)
                                )
            self.log_sigma_theta.append(torch.nn.Parameter(
                                            data=torch.randn(size=param.shape)\
                                                .to(self.base_device) - 4,
                                            requires_grad=param.requires_grad)
                                        )
            self.log_v_q.append(torch.nn.Parameter(
                                    data=torch.randn(size=param.shape).to(self.base_device) - 4,
                                    requires_grad=param.requires_grad)
                                )
            
        # NOTE: the learning rates for the inner-loop adaptation are defined in MetaBaseLearner

        self.gamma_p = torch.nn.Parameter(data=torch.tensor(float(gamma_p)).to(self.base_device))
        self.gamma_q = torch.nn.Parameter(data=torch.tensor(float(gamma_q)).to(self.base_device))

        # loading in meta optimizer 
        self.meta_lr = float(meta_lr)
        if optimizer_type == 'adam': 
            self.optimizer = torch.optim.Adam(params=self.meta_params_iter(), lr=self.meta_lr)
        else:
            logger.exception(f"Invalid optimizer type: {optimizer_type}")
            raise Exception(f"Invalid optimizer type: {optimizer_type}")

        # hyper-param for trading off ce-loss and kl-loss
        self.kl_weight = float(kl_weight)

        # number of steps to perform in the inner loop for conditioning the model/learning the 
        # meta-learning task
        self.num_conditioning_steps = int(num_conditioning_steps)
        self.num_learning_steps = int(num_learning_steps)

        # number of times to sample model weights
        self.num_model_samples = int(num_model_samples)
        
        # set flag to indicate if first-order approximation should be used (à la Reptile)
        if isinstance(use_first_order, str):
            use_first_order = eval(use_first_order)

        self.use_first_order = use_first_order

    ###### Helper functions ######

    def meta_params_iter(self):
        """ Returns an iterator over all of the meta parameters"""
        return itertools.chain(
                        self.mu_theta, self.log_sigma_theta, self.log_v_q, self.inner_layers_lr,
                        [self.gamma_p, self.gamma_q, self.classifier_lr],
                        self.retained_lm_head.values() if self.retain_lm_head else []
        )

    def get_task_init_kwargs(self, task_init_method, n_labels, **kwargs):
        """ 
        Override base implementation of this method to replace the model with the functional 
        model and also pass in the model params when the task head is initialized using  protomaml.

        Args:
            * task_init_method (str): Method for initializing the task head
            * n_labels (int): Number of labels defined by the task (i.e. classes)
        Returns:
            * init_kwargs (dict): Keyword arguments used by the initialization function 
        """

        init_kwargs = super().get_task_init_kwargs(task_init_method, n_labels, **kwargs)
        if 'protomaml' in task_init_method:
            init_kwargs['model'] = self.functional_model
            init_kwargs['params'] = self.mu_theta

        return init_kwargs

    ###### Model adaptation methods ######

    """
    Platipus relies on these methods to adapt and sample from the weight distributions 
    that it meta-learned. 
    """

    def _sample_adapted_params(self, data_batch, sampling_std, return_adapted_mean=False, 
                                device=None, **adaptation_kwargs): 
        """ 
        Helper method for sampling model weights that have been adapted to a batch of data. This 
        method samples the weights of the model self.num_model_samples times and terutns 

        Args: 
            * data_batch (dict): Batch of data for a forward pass through the model 
                (see run_inner_loop for information on the data structure)
            * sampling_std ([torch.Tensor]): The standard deviation to be used for sampling weights
                for each weight that is to-be meta-learned
            * return_adapted_mean (bool): Whether to also return the means from the adapted
                distribution over weights (defaults to False)
            * adaptation_kwargs: Kerword arguments to be passed into adapt_params_classification 
                (see this method for more information) 
        Returns: 
            * adapted_params ([torch.Tensor]): Each entry corresponds to the adapted means of
                the weight distribution (only returned if return_adapted_mean is set to True)
            * sampled_weights_list ([[torch.Tensor]]): A list of length self.num_model_samples
                where each entry corresponds to the generated list of tensor weights sampled from
                the model 
        """
        if device is None:
            # device is not None when we are doing multi-GPU training
            device = self.base_device

        adapted_params = self._adapt_params(data_batch, **adaptation_kwargs)


        sampled_weights_list = []

        for _ in range(self.num_model_samples):

            sampled_weights = [None] * len(adapted_params) 

            for i in range(len(sampled_weights)):
                if adapted_params[i].requires_grad: 
                    sampled_weights[i] = adapted_params[i] + \
                        torch.randn_like(input=adapted_params[i], device=device) * \
                            torch.exp(input=sampling_std[i])
                else: 
                    sampled_weights[i] = self.mu_theta[i]

            sampled_weights_list.append(sampled_weights)

        if return_adapted_mean:
            return (adapted_params, sampled_weights_list)
        else: 
            return sampled_weights_list  


    ###### Model training methods ######

    ### Multi Processing Helper Method
    def run_inner_loop_mp(self, rank, world_size, data_queue, loss_queue, step_optimizer, 
                          num_tasks_per_iteration):
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

            task_loss, (ce_loss, kl_loss) = ddp(self, support_batch, query_batch, device)

            task_loss = task_loss/num_tasks_per_iteration
            ce_loss = ce_loss/num_tasks_per_iteration
            kl_loss = kl_loss/num_tasks_per_iteration

            task_loss.backward()

            loss_queue.put([[task_loss.detach().item(),
                            ce_loss.detach().item(),
                            kl_loss.detach().item()]])

    ### Main Inner Training Loop 
    def run_inner_loop(self, support_batch, query_batch, device=None, *args, **kwargs): 
        """
        Implements steps 6-11 outlined in the algorithm 1 of the Platipus paper
        https://arxiv.org/pdf/1806.02817.pdf, i.e. the inner loop optimization step.
        
        Args: 
            * support_batch: A dictionary containing the following information for the support set
                * input_ids (torch.tensor): Input tensors of shape (N*K, max_seq_len)
                * input_target_idx (torch.tensor): Tensor indicating for each sample at what index
                    we apply the final classification layer 
                * label_ids (torch.tensor): Tensor of labels corresponding to masked out subword id
                * attention_mask (torch.tensor): Tensor indicating which tokens in input_ids are
                    not pad tokens
            * query_batch: Same as support_batch, but for the data of the query set 
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
        support_batch = move_to_device(support_batch, device)
        query_batch = move_to_device(query_batch, device)

        # Setting up LM head for task training
        if self.retain_lm_head:
            lm_head = self.retained_lm_head
        else: 
            init_kwargs = self.get_task_init_kwargs(self.lm_head_init_method, self.lm_head_n,
                                                    data_batch=support_batch, device=device)
            lm_head = TaskHead.initialize_task_head(task_type='classification',
                                                    method=self.lm_head_init_method,
                                                    init_kwargs=init_kwargs)

        # sampling theta after adapting model to query_batch
        mu_theta_query, theta_list = self._sample_adapted_params(
                                                        query_batch, 
                                                        sampling_std=self.log_v_q,
                                                        return_adapted_mean=True,
                                                        params=self.mu_theta,
                                                        lm_head_weights=lm_head,
                                                        learning_rate=self.gamma_q,
                                                        num_inner_steps=self.num_conditioning_steps,
                                                        clone_params=True,
                                                        device=device)
        

        ce_loss = 0.0
        for theta in theta_list: 
            # theta_list is the list of sampled model weights 
            # for each of these sampled weights (aka. thetas) we finetune the model on the 
            # support set and evaluate the resulting model

            # Make sure we don't change the LM head between different samples -- clone lm head
            adapted_lm_head = {key: torch.clone(param) for key, param in lm_head.items()}

            # adapting theta to the support set -> adapted params are phi
            phi = self._adapt_params(support_batch, 
                                     params=theta, 
                                     lm_head_weights=adapted_lm_head,
                                     learning_rate=self.inner_layers_lr,
                                     num_inner_steps=self.num_learning_steps,
                                     clone_params=False,
                                     optimize_classifier=True)

            # evaluating on the query batch using the adapted params phi  
            self.functional_model.eval()

            outputs = self.functional_model.forward(input_ids=query_batch['input_ids'],
                                                    attention_mask=query_batch['attention_mask'],
                                                    params=phi)

            _, sample_ce_loss = self._compute_task_loss(outputs, query_batch, adapted_lm_head, 
                                                        task_type='classification')
            

            ce_loss = ce_loss + sample_ce_loss
            self.functional_model.train()
        
        # average the cross-entropy loss by all the samples we generated
        ce_loss = ce_loss/len(theta_list)

        # computing KL loss (requires adapting mu_theta to the support set)
        
        # mu theta adapted to the support set (line 10 from algorithm 1) 
        mu_theta_support = self._adapt_params(support_batch, 
                                              params=self.mu_theta,
                                              lm_head_weights=lm_head,
                                              learning_rate=self.gamma_p,
                                              num_inner_steps=self.num_conditioning_steps,
                                              clone_params=True)

        kl_loss = kl_divergence_gaussians(p=[*mu_theta_query, *self.log_v_q],
                                          q=[*mu_theta_support, *self.log_sigma_theta])

        loss = ce_loss + self.kl_weight * kl_loss

        return loss, (ce_loss, self.kl_weight * kl_loss)


    ###### Model evaluation methods ######

    def run_finetuning(self, task_type, finetune_dataloader, adaptation_batch, n_labels, 
                       task_head_init_method, max_finetuning_batch_steps=-1, **kwargs): 
        """
        Creates a copy of the trained model parameters and continues to finetune these 
        parameters on a given dataset. Implements a  slightly adapted version of algorithm 2
        of the platipus paper https://arxiv.org/pdf/1806.02817.pdf. Note that when finetuning 
        and using the platipus method, we first need to sample weights for the model. We do so
        by adapting platipus to a batch of data from the finetune task that has been formatted
        to match the training LM task.
        
        Args: 
            * task_type (str): Type of task (e.g. 'classification')
            * finetune_dataloader (torch.data.Dataloader): The dataset for finetuning the model on
                is passed in as a dataloader (i.e. NLUDataloader)
            * adaptation_batch (dict): A batch of language data for initially adapting platipus 
                and sampling the weights of the model 
            * n_labels (int): The number of labels in the given finetuning task
            * task_head_init_method (str): Method for initializing task head 
            * max_finetuning_batch_steps (int): Optional maximum number of batch steps to take 
                for model finetuning 

        Returns:
            * inference_params dict containing: 
                * finetuned_params_list ([[nn.Parameter]]): List of finetuned model parameters
                    corresponding to self.num_model_samples samples of the model. 
                * task_head_weights_list ([dict]): List of task head weights for each of the
                    samples of model weights 
        """

        if not hasattr(self, "functional_model"):
            # NOTE: Edge case for if the model is only being evaluated without having 
            # been trained
            self.functionalize_model()

        self.functional_model.train()

        ### Sampling weights from mu_theta using the passed in adaptation_batch 
        # NOTE: the adaptation batch is in the same form as the batches of training used 
        # during meta training 

            
        if self.retain_lm_head:
            lm_head = self.retained_lm_head
        else:
            adaptation_batch = move_to_device(adaptation_batch, self.base_device)
            lm_init_kwargs = self.get_task_init_kwargs(self.lm_head_init_method, self.lm_head_n,
                                                    data_batch=adaptation_batch)
            lm_head = TaskHead.initialize_task_head(task_type='classification',
                                                    method=self.lm_head_init_method,
                                                    init_kwargs=lm_init_kwargs)

        theta_list = self._sample_adapted_params(adaptation_batch,
                                                 sampling_std=self.log_sigma_theta,
                                                 params=self.mu_theta,
                                                 lm_head_weights=lm_head,
                                                 learning_rate=self.gamma_p,
                                                 num_inner_steps=self.num_conditioning_steps,
                                                 clone_params=True,
                                                 evaluation_mode=True)

        ### Initializing the task head used for the downstream NLU task
        task_init_data_batch = move_to_device(next(iter(finetune_dataloader)), self.base_device)
        task_init_kwargs = self.get_task_init_kwargs(task_head_init_method, n_labels,
                                                     data_batch=task_init_data_batch)


        finetuned_params_list = []
        task_head_weights_list = []

        for theta in theta_list:
            task_head_weights = TaskHead.initialize_task_head(task_type=task_type,
                                                              method=task_head_init_method,
                                                              init_kwargs=task_init_kwargs)


            # detaching parameters from original computation graph to create new leaf variables
            finetuned_theta = []
            for p in theta:
                detached_p = p.detach()
                detached_p.requires_grad = p.requires_grad
                finetuned_theta.append(detached_p)

            finetuned_task_head_weights = {}
            for k, p in task_head_weights.items():
                detached_p = p.detach().clone()
                detached_p.requires_grad = True
                finetuned_task_head_weights[k] = detached_p

            finetune_params = itertools.chain(finetuned_theta,
                                            finetuned_task_head_weights.values())
            finetune_optimizer = torch.optim.Adam(params=finetune_params)

            for batch_idx, data_batch in enumerate(finetune_dataloader):
                data_batch = move_to_device(data_batch, self.base_device)
                finetune_optimizer.zero_grad()

                # run SGD on the finetuned theta parameters
                outputs = self.functional_model.forward(input_ids=data_batch['input_ids'],
                                                        attention_mask=data_batch['attention_mask'],
                                                        params=finetuned_theta)

                _, loss = self._compute_task_loss(outputs, data_batch, finetuned_task_head_weights, 
                                                task_type=task_type)

                loss.backward()
                finetune_optimizer.step()

                if max_finetuning_batch_steps > 0 and (batch_idx + 1) >= max_finetuning_batch_steps:
                    break

            finetuned_params_list.append(finetuned_theta)
            task_head_weights_list.append(finetuned_task_head_weights)

        inference_params = {
            "finetuned_params_list": finetuned_params_list, 
            "task_head_weights_list": task_head_weights_list
        }

        return inference_params


    def run_inference(self, task_type, inference_dataloader, finetuned_params_list,
                      task_head_weights_list, **kwargs):
        """ 
        This method is to be called after run_finetuning. 
        
        As the name suggests, this method runs inference on an NLU dataset for some task.
        The primary adaptation of the weights for a given task should occur in the
        run_finetuning method. 

        Args: 
            * task_type (str): Type of task (e.g. 'classification')
            * inference_dataloader (torch.data.Dataloader): The dataset for inference is passed
                in as a dataloader (in most cases this will be an NLUDataloader)
            * finetuned_params_list ([[nn.Parameter]]): List of finetuned model parameters
                corresponding to self.num_model_samples samples of the model. 
            * task_head_weights_list ([dict]): List of task head weights for each of the samples 
                of model weights 

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

                logits = 0.0
                loss = 0.0

                for finetuned_params, task_head_weights in zip(finetuned_params_list,
                                                               task_head_weights_list):

                    outputs = self.functional_model.forward(input_ids=data_batch['input_ids'],
                                                            attention_mask=\
                                                                data_batch['attention_mask'],
                                                            params=finetuned_params)

                    sample_logits, sample_loss = self._compute_task_loss(outputs, data_batch,
                                                                         task_head_weights,
                                                                         task_type=task_type)                    
                    
                    logits = logits + sample_logits
                    loss = loss + sample_loss
                
                logits = logits/len(finetuned_params_list)
                loss = loss/len(finetuned_params_list)

                predictions.extend(torch.argmax(logits, dim=-1).tolist())

                batch_size = logits.size(0)
                total_loss += loss.item() * batch_size # loss is averaged across batch
                total_samples += batch_size 

            total_loss /= total_samples

            self.functional_model.train()

        return (predictions, total_loss)
