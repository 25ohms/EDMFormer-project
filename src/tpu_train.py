#!/usr/bin/env python3
import argparse
import importlib
import os
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from torch import optim
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_cosine_schedule_with_warmup

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl


def load_checkpoint(checkpoint_dir: str, model, optimizer, scheduler):
    checkpoint_list = os.path.join(checkpoint_dir, "checkpoint")
    if not os.path.exists(checkpoint_list):
        return 0
    ckpt_name = open(checkpoint_list).readline().strip()
    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt.get("global_step", 0))


def save_checkpoint(checkpoint_dir: str, model, optimizer, scheduler, step: int):
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"model.ckpt-{step}.pt")
    xm.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "global_step": step,
        },
        ckpt_path,
    )
    if xm.is_master_ordinal():
        with open(os.path.join(checkpoint_dir, "checkpoint"), "w") as f:
            f.write(f"model.ckpt-{step}.pt")


def train_worker(index: int, args):
    device = xm.xla_device()
    torch.manual_seed(args.init_seed)

    hparams = OmegaConf.load(args.config)
    module = importlib.import_module("models." + args.model_name)
    Model = getattr(module, "Model")

    model = Model(hparams).to(device)
    optimizer = optim.Adam(model.parameters(), **hparams.optimizer)

    warmup_steps = hparams.warmup_steps
    total_steps = hparams.total_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    train_dataset = hydra.utils.instantiate(hparams.train_dataset)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True,
    )
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        **hparams.train_dataloader,
        collate_fn=train_dataset.collate_fn,
    )
    train_loader = pl.MpDeviceLoader(train_loader, device)

    global_step = 0
    if args.checkpoint_dir:
        if xm.is_master_ordinal():
            global_step = load_checkpoint(
                args.checkpoint_dir, model, optimizer, scheduler
            )
        step_tensor = torch.tensor(global_step, device=device, dtype=torch.int64)
        xm.all_reduce(xm.REDUCE_SUM, step_tensor)
        global_step = int(step_tensor.item())

    max_steps = args.max_steps or hparams.total_steps

    for epoch in range(args.max_epochs or hparams.args.max_epochs):
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            if global_step >= max_steps:
                break
            if batch is None:
                continue

            model.train()
            optimizer.zero_grad(set_to_none=True)
            _, loss, losses = model(batch)
            loss.backward()
            xm.optimizer_step(optimizer)
            scheduler.step()
            global_step += 1

            if global_step % args.log_interval == 0:
                xm.master_print(
                    f"epoch={epoch} step={global_step} loss={loss.item():.4f} "
                    f"loss_section={losses['loss_section'].item():.4f} "
                    f"loss_function={losses['loss_function'].item():.4f}"
                )

            if args.checkpoint_dir and global_step % args.save_interval == 0:
                xm.rendezvous("save_ckpt")
                if xm.is_master_ordinal():
                    save_checkpoint(
                        args.checkpoint_dir, model, optimizer, scheduler, global_step
                    )
                xm.rendezvous("save_ckpt_done")

        if global_step >= max_steps:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="TPU training entrypoint")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--init_seed", type=int, required=True)
    parser.add_argument("--model_name", type=str, default="SongFormer")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=800)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--num_cores", type=int, default=8)
    args = parser.parse_args()

    xmp.spawn(train_worker, args=(args,), nprocs=args.num_cores, start_method="fork")


if __name__ == "__main__":
    main()
