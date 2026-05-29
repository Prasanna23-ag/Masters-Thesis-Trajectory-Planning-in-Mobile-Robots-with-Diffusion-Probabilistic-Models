import socket

from diffuser.utils import watch

logbase= 'logs'

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

diffusion_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
]

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('discount', 'd'),
]


base = {
    'diffusion': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 32,
        'n_diffusion_steps': 100,
        'action_weight': 10,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 4, 8),
        'renderer': 'utils.MuJoCoRenderer',

        ## dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': True,
        'use_padding': True,
        'max_path_length': 1000,

        ## serialization
        'logbase': 'logs',
        'prefix': 'diffusion/',
        'exp_name': watch(diffusion_args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 1e6,    ####original code 1e6
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 1000,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda',
    },

    'values': {
        'model': 'models.ValueFunction',
        'diffusion': 'models.diffusion.ValueDiffusion',
        'horizon': 32,
        'n_diffusion_steps': 20,
        'dim_mults': (1, 2, 4, 8),
        'renderer': 'utils.MuJoCoRenderer',

        ## value-specific kwargs
        'discount': 0.997,
        'termination_penalty': -100,
        'normed': False,

        ## dataset
        'loader': 'datasets.ValueDataset',
        'normalizer': 'GaussianNormalizer',
        'preprocess_fns': [],
        'use_padding': True,
        'max_path_length': 1000,

        ## serialization
        'logbase': logbase,
        'prefix': 'values/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'value_l2',
        'n_train_steps': 200e3,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 0,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        # 'device': 'cuda',
        'seed': 0,
    },

    'plan': {
        'guide': 'sampling.ValueGuide',
        'policy': 'sampling.GuidedPolicy',
        'max_episode_length': 1000,
        'batch_size': 64,
        'preprocess_fns': [],
        # 'device': 'cuda',
        'seed': 0,

        ## sample_kwargs
        'n_guide_steps': 2,
        'scale': 0.1,
        't_stopgrad': 2,
        'scale_grad_by_std': True, 

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/',
        'exp_name': watch(args_to_watch),
        'vis_freq': 100,
        'max_render': 8,

        ## diffusion model
        'horizon': 32,
        'n_diffusion_steps': 20,

        ## value function
        'discount': 0.997,

        ## loading
        ## uncomment for half cheetah

        'diffusion_loadpath': 'f:logs/halfcheetah-medium-expert-v2/diffusion/defaults_H{horizon}_T{n_diffusion_steps}',
        'value_loadpath': 'f:logs/halfcheetah-medium-expert-v2/values/defaults_H{horizon}_T{n_diffusion_steps}_d{discount}',

        ## uncomment for maze2d

        # 'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}',
        # 'value_loadpath': 'f:values/H{horizon}_T{n_diffusion_steps}_d{discount}',

        'diffusion_epoch': 'latest',
        'value_epoch': 'latest',

        'verbose': True,
        'suffix': '0',
    },
}

#------------------------ overrides ------------------------#

## put environment-specific overrides here
hopper_medium_expert_v2 = {
    'plan': {
        'scale': 0.0001,
        't_stopgrad': 4,
    },
}


halfcheetah_medium_replay_v2 = halfcheetah_medium_v2 = halfcheetah_medium_expert_v2 = {
    'diffusion': {
        'horizon': 4,
        'dim_mults': (1, 4, 8),
        'attention': True,
    },
    'values': {
        'horizon': 4,
        'dim_mults': (1, 4, 8),
    },
    'plan': {
        'horizon': 4,
        'n_diffusion_steps': 20,
        'scale': 0.001,
        't_stopgrad': 4,
    },
}



maze2d_large_v1 = {
    'diffusion': {
        'horizon': 32,
        'n_diffusion_steps': 100,
    },
    'plan': {
        'horizon': 32,
        'n_diffusion_steps': 100,
    },
}
