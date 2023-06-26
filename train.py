import torch
from data import DiffSet
import pytorch_lightning as pl
from model import DiffusionModel
from torch.utils.data import DataLoader
import imageio
import glob
from tqdm import tqdm

if __name__ == '__main__':
    # Training hyperparameters
    diffusion_steps = 1000
    dataset_choice = "CIFAR"
    max_epoch = 10
    batch_size = 32

    # Loading parameters
    load_model = False
    load_version_num = 1

    # Code for optionally loading model
    pass_version = None
    last_checkpoint = None

    if load_model:
        pass_version = load_version_num
        last_checkpoint = glob.glob(
            f"./lightning_logs/{dataset_choice}/version_{load_version_num}/checkpoints/*.ckpt"
        )[-1]

    # Create datasets and data loaders
    train_dataset = DiffSet(True, dataset_choice)
    val_dataset = DiffSet(False, dataset_choice)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    # Create model and trainer
    if load_model:
        model = DiffusionModel.load_from_checkpoint(last_checkpoint, in_size=train_dataset.size*train_dataset.size, t_range=diffusion_steps, img_depth=train_dataset.depth)
    else:
        model = DiffusionModel(train_dataset.size*train_dataset.size, diffusion_steps, train_dataset.depth)

    # Load Trainer model
    tb_logger = pl.loggers.TensorBoardLogger(
        "lightning_logs/",
        name=dataset_choice,
        version=pass_version,
    )

    trainer = pl.Trainer(
        max_epochs=max_epoch, 
        log_every_n_steps=10, 
        #gpus=1, 
        #auto_select_gpus=True,
        #resume_from_checkpoint=last_checkpoint, 
        logger=tb_logger
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

    gif_shape = [3, 3]
    sample_batch_size = gif_shape[0] * gif_shape[1]
    n_hold_final = 10

    # Generate samples from denoising process
    gen_samples = []
    x = torch.randn((sample_batch_size, train_dataset.depth, train_dataset.size, train_dataset.size))
    sample_steps = torch.arange(model.t_range-1, 0, -1)
    for t in tqdm(sample_steps, desc="Sampling"):
        x = model.denoise_sample(x, t)
        if t % 50 == 0:
            gen_samples.append(x)
    for _ in range(n_hold_final):
        gen_samples.append(x)
    gen_samples = torch.stack(gen_samples, dim=0).moveaxis(2, 4).squeeze(-1)
    gen_samples = (gen_samples.clamp(-1, 1) + 1) / 2

    # Process samples and save as gif
    gen_samples = (gen_samples * 255).type(torch.uint8)
    gen_samples = gen_samples.reshape(-1, gif_shape[0], gif_shape[1], train_dataset.size, train_dataset.size, train_dataset.depth)

    def stack_samples(gen_samples, stack_dim):
        gen_samples = list(torch.split(gen_samples, 1, dim=1))
        for i in range(len(gen_samples)):
            gen_samples[i] = gen_samples[i].squeeze(1)
        return torch.cat(gen_samples, dim=stack_dim)

    gen_samples = stack_samples(gen_samples, 2)
    gen_samples = stack_samples(gen_samples, 2)

    imageio.mimsave(
        f"{trainer.logger.log_dir}/pred.gif",
        list(gen_samples.squeeze(-1)),
        format="GIF",
        duration = 200
    )