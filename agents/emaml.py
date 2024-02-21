"""
EMAMLConfig originated from PPO algorithm of Ray rllib: https://github.com/ray-project/ray/blob/master/rllib/algorithms/ppo/ppo.py

"""

import logging
from typing import List, Optional, Type, Union, TYPE_CHECKING
import torch
import numpy as np
import tree
import math
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.execution.rollout_ops import (
    standardize_fields,
    synchronous_parallel_sample,
)
from ray.rllib.execution.train_ops import (
    train_one_step,
    multi_gpu_train_one_step,
)
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.evaluation.metrics import collect_metrics
from ray.rllib.policy.sample_batch import concat_samples, DEFAULT_POLICY_ID, convert_ma_batch_to_sample_batch,SampleBatch
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY,LearnerInfoBuilder
from ray.rllib.utils.metrics import (
    LOAD_BATCH_TIMER,
    NUM_AGENT_STEPS_TRAINED_THIS_ITER,
    NUM_AGENT_STEPS_TRAINED,
    NUM_ENV_STEPS_TRAINED,
    NUM_ENV_STEPS_TRAINED_THIS_ITER,
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
    SAMPLE_TIMER,
    LEARN_ON_BATCH_TIMER,
    ALL_MODULES,
)
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.utils.schedules.scheduler import Scheduler
from ray.rllib.utils.typing import ResultDict
from ray.util.debug import log_once

if TYPE_CHECKING:
    from ray.rllib.core.learner.learner import Learner


logger = logging.getLogger(__name__)

LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY = "vf_loss_unclipped"
LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY = "vf_explained_var"
LEARNER_RESULTS_KL_KEY = "mean_kl_loss"
LEARNER_RESULTS_CURR_KL_COEFF_KEY = "curr_kl_coeff"
LEARNER_RESULTS_CURR_ENTROPY_COEFF_KEY = "curr_entropy_coeff"


class EMAMLConfig(AlgorithmConfig):
    """Defines a configuration class from which a PPO Algorithm can be built.

    .. testcode::

        from ray.rllib.algorithms.ppo import PPOConfig
        config = PPOConfig()
        config = config.training(gamma=0.9, lr=0.01, kl_coeff=0.3,
            train_batch_size=128)
        config = config.resources(num_gpus=0)
        config = config.rollouts(num_rollout_workers=1)

        # Build a Algorithm object from the config and run 1 training iteration.
        algo = config.build(env="CartPole-v1")
        algo.train()

    .. testcode::

        from ray.rllib.algorithms.ppo import PPOConfig
        from ray import air
        from ray import tune
        config = PPOConfig()
        # Print out some default values.

        # Update the config object.
        config.training(
            lr=tune.grid_search([0.001 ]), clip_param=0.2
        )
        # Set the config object's env.
        config = config.environment(env="CartPole-v1")

        # Use to_dict() to get the old-style python config dict
        # when running with tune.
        tune.Tuner(
            "PPO",
            run_config=air.RunConfig(stop={"training_iteration": 1}),
            param_space=config.to_dict(),
        ).fit()

    .. testoutput::
        :hide:

        ...
    """

    def __init__(self, algo_class=None):
        """Initializes a EMAMLConfig instance."""
        super().__init__(algo_class=algo_class or EMAML)

        # fmt: off
        # __sphinx_doc_begin__
        self.lr_schedule = None
        self.lr = 1e-3
        self.train_batch_size = 2000

        # EMAML specific settings:
        self.use_gae = True
        self.lambda_ = 1.0
        self.kl_coeff = 0.0005
        self.vf_loss_coeff = 0.5
        self.entropy_coeff = 0.0
        self.clip_param = 0.3
        self.vf_clip_param = 10.0
        self.grad_clip = None
        self.kl_target = 0.01
        self.inner_adaptation_steps = 1
        self.maml_optimizer_steps = 5
        self.inner_lr = 1e-3
        self.use_meta_env = True

        self.use_critic = True
        self.use_kl_loss = True
        self.sgd_minibatch_size = 128
        # Simple logic for now: If None, use `train_batch_size`.
        self.mini_batch_size_per_learner = None
        self.num_sgd_iter = 30
        self.shuffle_sequences = True
        self.entropy_coeff_schedule = None

        # Override some of AlgorithmConfig's default values with E-MAML-specific values.
        self.num_rollout_workers = 4
        self.rollout_fragment_length = 100
        self.model["vf_share_layers"] = False
        self.create_env_on_local_worker = True
        # __sphinx_doc_end__
        # fmt: on

        # Deprecated keys.
        self.vf_share_layers = DEPRECATED_VALUE

        self.exploration_config = {
            # The Exploration class to use. In the simplest case, this is the name
            # (str) of any class present in the `rllib.utils.exploration` package.
            # You can also provide the python class directly or the full location
            # of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
            # EpsilonGreedy").
            "type": "StochasticSampling",
            # Add constructor kwargs here (if any).
        }

    @override(AlgorithmConfig)
    def training(
        self,
        *,
        lr_schedule: Optional[List[List[Union[int, float]]]] = NotProvided,
        use_critic: Optional[bool] = NotProvided,
        use_gae: Optional[bool] = NotProvided,
        lambda_: Optional[float] = NotProvided,
        use_kl_loss: Optional[bool] = NotProvided,
        kl_coeff: Optional[float] = NotProvided,
        kl_target: Optional[float] = NotProvided,
        mini_batch_size_per_learner: Optional[int] = NotProvided,
        shuffle_sequences: Optional[bool] = NotProvided,
        vf_loss_coeff: Optional[float] = NotProvided,
        entropy_coeff: Optional[float] = NotProvided,
        entropy_coeff_schedule: Optional[List[List[Union[int, float]]]] = NotProvided,
        clip_param: Optional[float] = NotProvided,
        vf_clip_param: Optional[float] = NotProvided,
        grad_clip: Optional[float] = NotProvided,
        inner_adaptation_steps: Optional[int] = NotProvided,
        maml_optimizer_steps: Optional[int] = NotProvided,
        inner_lr: Optional[float] = NotProvided,
        use_meta_env: Optional[bool] = NotProvided,
        # Deprecated.
        vf_share_layers=DEPRECATED_VALUE,
        **kwargs,
    ) -> "EMAMLConfig":
        """Sets the training related configuration.

        Args:
            lr_schedule: Learning rate schedule. In the format of
                [[timestep, lr-value], [timestep, lr-value], ...]
                Intermediary timesteps will be assigned to interpolated learning rate
                values. A schedule should normally start from timestep 0.
            use_critic: Should use a critic as a baseline (otherwise don't use value
                baseline; required for using GAE).
            use_gae: If true, use the Generalized Advantage Estimator (GAE)
                with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
            lambda_: The GAE (lambda) parameter.
            use_kl_loss: Whether to use the KL-term in the loss function.
            kl_coeff: Initial coefficient for KL divergence.
            kl_target: Target value for KL divergence.
            mini_batch_size_per_learner: Only use if new API stack is enabled.
                The mini batch size per Learner worker. This is the
                batch size that each Learner worker's training batch (whose size is
                `s`elf.train_batch_size_per_learner`) will be split into. For example,
                if the train batch size per Learner worker is 4000 and the mini batch
                size per Learner worker is 400, the train batch will be split into 10
                equal sized chunks (or "mini batches"). Each such mini batch will be
                used for one SGD update. Overall, the train batch on each Learner
                worker will be traversed `self.num_sgd_iter` times. In the above
                example, if `self.num_sgd_iter` is 5, we will altogether perform 50
                (10x5) SGD updates per Learner update step.
            sgd_minibatch_size: Total SGD batch size across all devices for SGD.
                This defines the minibatch size within each epoch. Deprecated on the
                new API stack (use `mini_batch_size_per_learner` instead).
            num_sgd_iter: Number of SGD iterations in each outer loop (i.e., number of
                epochs to execute per train batch).
            shuffle_sequences: Whether to shuffle sequences in the batch when training
                (recommended).
            vf_loss_coeff: Coefficient of the value function loss. IMPORTANT: you must
                tune this if you set vf_share_layers=True inside your model's config.
            entropy_coeff: Coefficient of the entropy regularizer.
            entropy_coeff_schedule: Decay schedule for the entropy regularizer.
            clip_param: The PPO clip parameter.
            vf_clip_param: Clip param for the value function. Note that this is
                sensitive to the scale of the rewards. If your expected V is large,
                increase this.
            grad_clip: If specified, clip the global norm of gradients by this amount.

        Returns:
            This updated AlgorithmConfig object.
        """
        # Pass kwargs onto super's `training()` method.
        super().training(**kwargs)

        if use_critic is not NotProvided:
            self.use_critic = use_critic
            # TODO (Kourosh) This is experimental.
            #  Don't forget to remove .use_critic from algorithm config.
        if use_gae is not NotProvided:
            self.use_gae = use_gae
        if lambda_ is not NotProvided:
            self.lambda_ = lambda_
        if use_kl_loss is not NotProvided:
            self.use_kl_loss = use_kl_loss
        if kl_coeff is not NotProvided:
            self.kl_coeff = kl_coeff
        if kl_target is not NotProvided:
            self.kl_target = kl_target
        if mini_batch_size_per_learner is not NotProvided:
            self.mini_batch_size_per_learner = mini_batch_size_per_learner
        if shuffle_sequences is not NotProvided:
            self.shuffle_sequences = shuffle_sequences
        if vf_loss_coeff is not NotProvided:
            self.vf_loss_coeff = vf_loss_coeff
        if entropy_coeff is not NotProvided:
            self.entropy_coeff = entropy_coeff
        if clip_param is not NotProvided:
            self.clip_param = clip_param
        if vf_clip_param is not NotProvided:
            self.vf_clip_param = vf_clip_param
        if grad_clip is not NotProvided:
            self.grad_clip = grad_clip
        if inner_adaptation_steps is not NotProvided:
            self.inner_adaptation_steps = inner_adaptation_steps
        if maml_optimizer_steps is not NotProvided:
            self.maml_optimizer_steps = maml_optimizer_steps
        if inner_lr is not NotProvided:
            self.inner_lr = inner_lr
        if use_meta_env is not NotProvided:
            self.use_meta_env = use_meta_env

        # TODO (sven): Remove these once new API stack is only option for PPO.
        if lr_schedule is not NotProvided:
            self.lr_schedule = lr_schedule
        if entropy_coeff_schedule is not NotProvided:
            self.entropy_coeff_schedule = entropy_coeff_schedule

        return self

    @override(AlgorithmConfig)
    def validate(self) -> None:
        # Call super's validation method.
        super().validate()

        # Synchronous sampling, on-policy/PPO algos -> Check mismatches between
        # `rollout_fragment_length` and `train_batch_size_per_learner` to avoid user
        # confusion.
        # TODO (sven): Make rollout_fragment_length a property and create a private
        #  attribute to store (possibly) user provided value (or "auto") in. Deprecate
        #  `self.get_rollout_fragment_length()`.

        # SGD minibatch size must be smaller than train_batch_size (b/c
        # we subsample a batch of `sgd_minibatch_size` from the train-batch for
        # each `num_sgd_iter`).

        # Episodes may only be truncated (and passed into PPO's
        # `postprocessing_fn`), iff generalized advantage estimation is used
        # (value function estimate at end of truncated episode to estimate
        # remaining value).
        if (
            not self.in_evaluation
            and self.batch_mode == "truncate_episodes"
            and not self.use_gae
        ):
            raise ValueError(
                "Episode truncation is not supported without a value "
                "function (to estimate the return at the end of the truncated"
                " trajectory). Consider setting "
                "batch_mode=complete_episodes."
            )

        # Entropy coeff schedule checking.
        if self._enable_new_api_stack:
            if self.entropy_coeff_schedule is not None:
                raise ValueError(
                    "`entropy_coeff_schedule` is deprecated and must be None! Use the "
                    "`entropy_coeff` setting to setup a schedule."
                )
            Scheduler.validate(
                fixed_value_or_schedule=self.entropy_coeff,
                setting_name="entropy_coeff",
                description="entropy coefficient",
            )
        if isinstance(self.entropy_coeff, float) and self.entropy_coeff < 0.0:
            raise ValueError("`entropy_coeff` must be >= 0.0")
        
class EMAML(Algorithm):
    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return EMAMLConfig()
    
    @classmethod
    @override(Algorithm)
    def get_default_policy_class(
        cls, config: AlgorithmConfig
    ) -> Optional[Type[Policy]]:
        if config["framework"] == "torch":
            from emaml_policy import MAMLTorchPolicy
            return MAMLTorchPolicy
        else:
            raise Exception
        
    @override(Algorithm)
    def training_step(self) -> ResultDict:

        local_worker = self.workers.local_worker()

        # 0. Sample tasks and assign
        worker_ids = self.workers.healthy_worker_ids()
        tasks = local_worker.env.sample_tasks(len(worker_ids))
        task_map = dict(zip(worker_ids,tasks))

        def set_tasks(e, ctx):
            e.init_adaptation()
            if ctx.worker_index in worker_ids:
                e.set_task(task_map[ctx.worker_index])
        
        self.workers.foreach_env_with_context(set_tasks)
        logger.debug("%d workers got %s tasks.", len(worker_ids), tasks.__str__())
        num_devices = int(math.ceil(self.config["num_gpus"] or 1))

        max_agent_steps = self.config.train_batch_size
        # [Inner Loop]
        def inner_loop(worker_id: int, worker: RolloutWorker):
            # 1. rollout episodes with previous policy (\phi_i)
            cur_steps = 0
            adapt_batch = []
            while cur_steps < max_agent_steps:
                sample = worker.sample()
                adapt_batch.append(sample)
                cur_steps+=sample.count
                
            adapt_batch = concat_samples(adapt_batch)

            # into sample batch and standardize adv
            adapt_batch = convert_ma_batch_to_sample_batch(adapt_batch)
            adapt_batch = standardize_fields(adapt_batch, ["advantages"])
            #adapt_batch.decompress_if_needed()

            # 2. preload and update policy individually (\phi_i -> \phi_i+1)
            worker.policy_map[DEFAULT_POLICY_ID].load_batch_into_buffer(adapt_batch, buffer_index=0)
            adapt_result = worker.foreach_policy(lambda p,pid: p.learn_on_loaded_batch())[0]
            return adapt_result, adapt_batch

        buf = []
        split = []
        with self._timers[LEARN_ON_BATCH_TIMER]:
            for i in range(self.config['inner_adaptation_steps']):
            
                rs = self.workers.foreach_worker_with_id(inner_loop, local_worker=False)
                split_list = []
                for r,b in rs:
                    #logger.debug("Sampled %d per worker", b.count)
                    #print(b.zero_padded, b.max_seq_len, b.time_major)
                    buf.append(b)
                    split_list.append(b.count)
                logger.debug("Adaptation %d", i+1)
                split.append(split_list)

        inner_metrics = collect_metrics(self.workers)        
        # 3. Change into post-adaptation mode, and rollout trajectory
        # Turning env into post-adapt mode is "very" ARCLE specific. Be careful!
        self.workers.foreach_env(lambda e: e.post_adaptation()) 

        logger.debug("Changed into Post-adaptation mode.")

        def outer_sample(worker):
            cur_steps = 0
            adapt_batch = []
            while cur_steps < max_agent_steps:
                sample = worker.sample(explore=False)
                adapt_batch.append(sample)
                cur_steps+=sample.count
                
            adapt_batch = concat_samples(adapt_batch)
            return adapt_batch

        with self._timers[SAMPLE_TIMER]:
            #post_batches = self.workers.foreach_worker(lambda w: w.sample(),local_worker=False)
            post_batches = self.workers.foreach_worker(outer_sample, local_worker=False)

        inner_test_metrics = collect_metrics(self.workers)
        # Convert multi-agent batches into single experiences
        
        split_list = []
        once_successful = []
        successful_batches = [] 
        for batch in post_batches:
            b = convert_ma_batch_to_sample_batch(batch)
            rewards = b[SampleBatch.REWARDS]
            if rewards.max() > 0:
                once_successful.append(True)
                successful_batches.append(b)
            else:
                once_successful.append(False)
                successful_batches.append(None)
            b = standardize_fields(b, ["advantages"])
            self._counters[NUM_AGENT_STEPS_SAMPLED] += b.agent_steps()
            self._counters[NUM_ENV_STEPS_SAMPLED] += b.env_steps()
            #print(b.zero_padded, b.max_seq_len, b.time_major)
            for k in b:
                if k==SampleBatch.EPS_ID:
                    b[k] = torch.tensor(b[k].astype(np.int64))
                elif k==SampleBatch.INFOS:
                    pass
                else:
                    b[k] = torch.tensor(b[k])
            buf.append(b)
            split_list.append(b.count)
            
        split.append(split_list)
        #logger.debug(split)
        train_batch = concat_samples(buf)
        train_batch["split"] = np.array(split) # it is neccesary in MAMLPolicy
        
        



        # 4. [Outer] Meta-update step (\theta -> \theta')
        outer_info_builder = LearnerInfoBuilder()
        
        for i in range(self.config.maml_optimizer_steps):
            # TODO: This is MAML update. let's add initial batches to modify as a E-MAML.
            train_results = local_worker.learn_on_batch(train_batch)
            outer_info_builder.add_learn_on_batch_results(train_results)

        outer_train_info = outer_info_builder.finalize()

        # 5. Sync weights and KL divergences of remote policies with meta-learned policy (\phi_0 <- \theta')
        with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
            self.workers.sync_weights()

        # --- From PPO Algorithm (KL Sync part) ---
        # For each policy: Update innerloop KL scale and warn about possible issues
        for policy_id, policy_info in train_results.items():
            # Update KL loss with dynamic scaling
            # `update_kls` for multiple kl divs 
            kl_divergence = policy_info[LEARNER_STATS_KEY].get("inner_kl")
            self.get_policy(policy_id).update_kls(kl_divergence)

            # Warn about excessively high value function loss
            scaled_vf_loss = (
                self.config.vf_loss_coeff * policy_info[LEARNER_STATS_KEY]["vf_loss"]
            )
            policy_loss = policy_info[LEARNER_STATS_KEY]["policy_loss"]
            if (
                log_once("ppo_warned_lr_ratio")
                and self.config.get("model", {}).get("vf_share_layers")
                and scaled_vf_loss > 100
            ):
                logger.warning(
                    "The magnitude of your value function loss for policy: {} is "
                    "extremely large ({}) compared to the policy loss ({}). This "
                    "can prevent the policy from learning. Consider scaling down "
                    "the VF loss by reducing vf_loss_coeff, or disabling "
                    "vf_share_layers.".format(policy_id, scaled_vf_loss, policy_loss)
                )
            """ # Warn about bad clipping configs.
            train_batch.policy_batches[policy_id].set_get_interceptor(None)
            mean_reward = train_batch.policy_batches[policy_id]["rewards"].mean()
            if (
                log_once("ppo_warned_vf_clip")
                and mean_reward > self.config.vf_clip_param
            ):
                self.warned_vf_clip = True
                logger.warning(
                    f"The mean reward returned from the environment is {mean_reward}"
                    f" but the vf_clip_param is set to {self.config['vf_clip_param']}."
                    f" Consider increasing it for policy: {policy_id} to improve"
                    " value function convergence."
                )
 """
        self._counters[NUM_ENV_STEPS_TRAINED] += train_batch.count
        self._counters[NUM_AGENT_STEPS_TRAINED] += train_batch.agent_steps()

        return {
                'sampled_tasks':tasks,
                'once_successful': once_successful,
                'successful_batches': successful_batches,
                'inner_metrics':inner_metrics, 
                'inner_test_metrics':inner_test_metrics, 
                'meta_train_info': outer_train_info
            }

