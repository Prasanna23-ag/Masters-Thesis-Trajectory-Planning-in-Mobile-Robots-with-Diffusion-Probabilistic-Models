import json
import numpy as np
from os.path import join
import pdb
import diffuser.environments  # triggers your register() call
import matplotlib.pyplot as plt

from diffuser.guides.policies import Policy
import diffuser.utils as utils
import diffuser.datasets as datasets
from torch.utils.tensorboard import SummaryWriter


# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#


class Parser(utils.Parser):
    dataset: str = "birrt-dataset-v0"
    config: str = "config.maze2d_hl"


args = Parser().parse_args("diffusion")

# -----------------------------------------------------------------------------#
# ---------------------------------- dataset ----------------------------------#
# -----------------------------------------------------------------------------#

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, "dataset_config.pkl"),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
    jump=args.jump,
    jump_action=args.jump_action,
)

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, "render_config.pkl"),
    env=args.dataset,
)

dataset = dataset_config()
renderer = render_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim * args.jump
if args.jump_action == "none":
    action_dim = 0


# -----------------------------------------------------------------------------#
# ------------------------------ model & trainer ------------------------------#
# -----------------------------------------------------------------------------#

model_config = utils.Config(
    args.model,
    savepath=(args.savepath, "model_config.pkl"),
    horizon=args.horizon // args.jump,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim=args.dim,
    dim_mults=args.dim_mults,
    kernel_size=args.kernel_size,
    device=args.device,
    upsample_k=args.upsample_k,
    downsample_k=args.downsample_k,
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, "diffusion_config.pkl"),
    horizon=args.horizon // args.jump,
    condition=args.condition,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    ## loss weighting
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=args.device,
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, "trainer_config.pkl"),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
    n_samples=args.n_samples,
)

# -----------------------------------------------------------------------------#
# -------------------------------- instantiate --------------------------------#
# -----------------------------------------------------------------------------#

model = model_config()

diffusion = diffusion_config(model)

trainer = trainer_config(diffusion, dataset, renderer)


# -----------------------------------------------------------------------------#
# ------------------------ test forward & backward pass -----------------------#
# -----------------------------------------------------------------------------#

utils.report_parameters(model)

print("Testing forward...", end=" ", flush=True)
batch = utils.batchify(dataset[0])
loss, _ = diffusion.loss(*batch)
loss.backward()
print("✓")


# -----------------------------------------------------------------------------#
# --------------------------------- main loop ---------------------------------#
# -----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)
eval_sample_n = 3

train_writer = SummaryWriter(log_dir=args.savepath + "-train")

# ---------------used for turtlebot env-------------#
total_steps = 0
losses, a0_losses, a_losses, s_losses, t_losses = [], [], [], [], []
for i in range(n_epochs):
    print(f"Epoch {i} / {n_epochs} | {args.savepath}")
    avg_loss = trainer.train(n_train_steps=args.n_steps_per_epoch, writer=train_writer)
    losses.append(avg_loss['loss'])         ## total loss
    a0_losses.append(avg_loss['a0_loss'])   ## first action loss
    a_losses.append(avg_loss['a_loss'])     ## action loss (all actions)
    s_losses.append(avg_loss['s_loss'])     ## state loss
    t_losses.append(avg_loss['t_loss'])     ## terminal loss

    total_steps += args.n_steps_per_epoch

    #  Render every 10k steps
    if total_steps % 1000 == 0:
        print(f"[ rendering ] Step {total_steps} — generating reference image")
        trainer.render_reference(trainer.n_reference, step=total_steps)

###### Plot all loss components ######
plt.figure(figsize=(10,6))
plt.plot(losses, label="Total Loss")
plt.plot(a0_losses, label="a0_loss")
plt.plot(a_losses, label="a_loss")
plt.plot(s_losses, label="s_loss")
plt.plot(t_losses, label="t_loss")

plt.xlabel("Epoch")
plt.ylabel("Loss Value")
plt.title("Training Loss Components")
plt.legend()
plt.grid(True)

plt.savefig(join(args.savepath, "all_losses_curve.png"))
plt.close()

