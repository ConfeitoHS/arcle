"""
EMAMLConfig originated from PPO algorithm of Ray rllib: https://github.com/ray-project/ray/blob/master/rllib/algorithms/ppo/ppo.py

"""

import logging
from typing import List, Optional, Type, Union, TYPE_CHECKING

import numpy as np
import tree

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
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
    SAMPLE_TIMER,
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
        self.train_batch_size = 100

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
        self.inner_lr = 0.1
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
        self.num_rollout_workers = 2
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
        sgd_minibatch_size: Optional[int] = NotProvided,
        num_sgd_iter: Optional[int] = NotProvided,
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
        if sgd_minibatch_size is not NotProvided:
            self.sgd_minibatch_size = sgd_minibatch_size
        if num_sgd_iter is not NotProvided:
            self.num_sgd_iter = num_sgd_iter
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
        self.validate_train_batch_size_vs_rollout_fragment_length()

        # SGD minibatch size must be smaller than train_batch_size (b/c
        # we subsample a batch of `sgd_minibatch_size` from the train-batch for
        # each `num_sgd_iter`).
        if (
            not self._enable_new_api_stack
            and self.sgd_minibatch_size > self.train_batch_size
        ):
            raise ValueError(
                f"`sgd_minibatch_size` ({self.sgd_minibatch_size}) must be <= "
                f"`train_batch_size` ({self.train_batch_size}). In PPO, the train batch"
                f" will be split into {self.sgd_minibatch_size} chunks, each of which "
                f"is iterated over (used for updating the policy) {self.num_sgd_iter} "
                "times."
            )
        elif self._enable_new_api_stack:
            mbs = self.mini_batch_size_per_learner or self.sgd_minibatch_size
            tbs = self.train_batch_size_per_learner or self.train_batch_size
            if mbs > tbs:
                raise ValueError(
                    f"`mini_batch_size_per_learner` ({mbs}) must be <= "
                    f"`train_batch_size_per_learner` ({tbs}). In PPO, the train batch"
                    f" will be split into {mbs} chunks, each of which is iterated over "
                    f"(used for updating the policy) {self.num_sgd_iter} times."
                )

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
            
            return EMAMLTorchPolicy
        else:
            raise Exception
            return None
        
    @override(Algorithm)
    def training_step(self) -> ResultDict:

        # 0. Sample tasks and assign
        worker_ids = self.workers.healthy_worker_ids()
        tasks = self.workers.local_worker().env.sample_tasks(len(worker_ids))
        task_map = dict(zip(worker_ids,tasks))
        self.workers.foreach_env_with_context(lambda e,ctx: e.set_task(task_map[ctx.worker_index]))


        # 1. rollout episodes with exploratory initial policy (\theta = \phi_0)
        train_batches = synchronous_parallel_sample(worker_set=self.workers, max_agent_steps=self.config.train_batch_size)
        train_batches = train_batches.as_multi_agent()


        # 2. [Inner] Update policy in each worker individually (\phi_0 -> \phi_N)
        inner_steps = self.config['inner_adaptation_steps']
        def inner_loop(policy: Policy, policy_id : str):
            #we chould use this grad_info to meta-update
            info = []
            for it in range(inner_steps):
                grad_info = policy.learn_on_batch(train_batches[policy_id])
                info.append(grad_info)
            return info
        
        standardize_fields(train_batches, ["advantages"])
        inner_train_info = self.workers.foreach_policy(inner_loop)

        # 3. Change into post-adaptation mode, and rollout trajectory
        self.workers.foreach_env(lambda e: e.post_adaptation()) # it is "very" ARCLE Specific. Be careful!
        test_batches = synchronous_parallel_sample(worker_set=self.workers, max_agent_steps=self.config.train_batch_size)
        test_batches = test_batches.as_multi_agent() # Is it really needed?
        
        # 4. [Outer] Meta-update (\theta -> \theta')
        for i in range(self.config.maml_optimizer_steps):
            meta_grad_info = self.workers.local_worker().learn_on_batch(test_batches)
        
        # 5. Sync weights of remote policies with meta-learned local policy (\phi_0 <- \theta')
        self.workers.sync_weights()
        
        return {}
        
        
        