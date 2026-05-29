import diffuser.utils as utils
import os
import importlib
importlib.reload(utils.colab)


class Args:
  loadpath = 'Thesis/diffuser/logs/pretrained/hopper-medium-expert-v2/diffusion/H128_T100'
  diffusion_epoch = 'latest'
  n_samples = 4
  device = 'cuda:0'
    
args = Args()
print(args.loadpath)

#config_path = os.path.join(args.loadpath, 'dataset_config.pkl')
#print("Looking for:", config_path)
#print("Exists?", os.path.exists(config_path))

diffusion_experiment = utils.load_diffusion(args.loadpath, epoch=args.diffusion_epoch)

dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer
model = diffusion_experiment.trainer.ema_model


env = dataset.env
obs = env.reset()


observations = utils.colab.run_diffusion(model, dataset, obs, args.n_samples, args.device)
print(observations.shape)


sample = observations[-1]
#utils.colab.savebase = os.path.join(os.getcwd(), 'test')  # You can name it anything you like
#os.makedirs(utils.colab.savebase, exist_ok=True) 
utils.colab.show_sample(renderer, sample)
