from typing import List, SupportsFloat, SupportsInt, Tuple
import logging, random, pickle, os

import numpy as np
import torch
import wandb

import ray
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print

from arcle.envs import O2ARCv2Env
from arcle.loaders import ARCLoader, Loader
from arcle.wrappers import BBoxWrapper

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from gymnasium.wrappers.flatten_observation import FlattenObservation

from emaml import EMAML, EMAMLConfig
from env import CustomO2ARCEnv, FilterO2ARC


logging.basicConfig(level=logging.DEBUG)


seed = 8182
deterministic = True
run_name = f"maml-crop-seed{seed}"

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if deterministic:
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    #configs
    n_cpus = 10
    n_envs = 10
    n_inner_steps = 20
    n_outer_steps = 5
    n_batch = 1000

    l_rollout = 100
    l_episode = 100
    # so total ep in one env&step is n_batch/l_episode

    inner_lr = 1e-3
    outer_lr = 1e-4
    discount = 0.90
    vf_loss_coeff = 0.1

    grad_clip = 10
    fc_hiddens = [1024,1024,512,512,256,128]

    ray.init()
    def env_creator(config):
        env = CustomO2ARCEnv(max_trial=127)
        env = BBoxWrapper(env)
        env = FilterO2ARC(env)
        env = FlattenObservation(env)
        env = TimeLimit(env, max_episode_steps=l_episode)
        return env

    register_env("O2ARCBBoxEnv", env_creator)
    wandb.login(key = '6411bae5ab591c488ef3a5ef3f1c6febbb9b768e')
    wandb.init(
        # set the wandb project where this run will be logged
        project="arcle-maml",
        name=run_name,
        # track hyperparameters and run metadata
        config={
        "seed": seed,
        "vf_loss_coeff": vf_loss_coeff,
        "batch_size":n_batch, "rollout_fragment_len":l_rollout, "max_episode_steps":l_episode,
        "tasks_parallel":n_envs, 
        "inner_steps": n_inner_steps, "outer_steps": n_outer_steps,
        "inner_lr": inner_lr, "outer_lr" : outer_lr,
        "architecture": f"MLP{fc_hiddens}",
        "dataset": "FullARCTrainset",
        }
    )

    config = (EMAMLConfig()
            .resources(num_gpus=1, num_cpus_per_worker=n_cpus/n_envs)
            .rollouts(num_rollout_workers=n_envs, rollout_fragment_length=l_rollout)
            .environment(env="O2ARCBBoxEnv")
            .training(gamma=discount, 
                        lr=outer_lr, inner_lr=inner_lr,
                        inner_adaptation_steps=n_inner_steps, maml_optimizer_steps=n_outer_steps,
                        grad_clip=grad_clip, train_batch_size=n_batch, 
                        vf_loss_coeff= vf_loss_coeff, model={
                "fcnet_hiddens": fc_hiddens,
                "fcnet_activation": "tanh",
            })
            
            )
    algo = config.build()

    epoch = 1
    tot_tasks = len(ARCLoader().data)
    tasks_covered = [0]*tot_tasks
    succeed = [0]*tot_tasks
    succeed_hist = []
    while True:
        res = algo.train()
        learn_info = res["info"]["learner"]
        inner_metrics = learn_info["inner_metrics"]
        inner_test_metrics = learn_info["inner_test_metrics"]
        meta_train_info = learn_info["meta_train_info"]["default_policy"]["default_policy"]["learner_stats"]
        print(learn_info["sampled_tasks"])
        for _t in learn_info["sampled_tasks"]:
            print(type(_t))
        cur_succeed_hist = []

        for _t, _s, _sb in zip(learn_info["sampled_tasks"],learn_info["once_successful"],learn_info["successful_batches"]):
            tasks_covered[_t]+=1
            if _s:
                succeed[_t]+=1
                cur_succeed_hist.append({"task_idx":_t, "batch":_sb})
                os.makedirs(f'./ckpts/{run_name}/successful',exist_ok=True)
                with open(f'./ckpts/{run_name}/successful/epoch{epoch}_{_t}.pickle', mode='wb') as fp:
                    pickle.dump({"task_idx":_t, "batch":_sb},fp)
        succeed_hist.append(cur_succeed_hist)
        wandb.log({
            'num_env_steps_trained': res['num_env_steps_trained'],
            'num_healthy_workers': res['num_healthy_workers'],

            'outer_kl_loss': meta_train_info['kl_loss'],
            'outer_policy_loss': meta_train_info['policy_loss'],
            'outer_vf_loss': meta_train_info['vf_loss'],
            'outer_total_loss': meta_train_info['total_loss'],

            'adapt_eprewmax': inner_metrics['episode_reward_max'],
            'adapt_eprewmean': inner_metrics['episode_reward_mean'],
            'adapt_eprewmin': inner_metrics['episode_reward_min'],

            'post_eprewmax': inner_test_metrics['episode_reward_max'],
            'post_eprewmean': inner_test_metrics['episode_reward_mean'],
            'post_eprewmin': inner_test_metrics['episode_reward_min'],
            
            'num_covered_tasks': (np.array(tasks_covered)>0).sum(),
            'num_succeed_tasks': (np.array(succeed)>0).sum(),
            #'cur_succeed_hist': cur_succeed_hist
        }, step=res['training_iteration'])

        
        if epoch%10==0:
            save_res = algo.save(f'./agents/ckpts/{run_name}/epoch{epoch}/')
        epoch+=1
        
        
