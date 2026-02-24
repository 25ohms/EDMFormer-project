#!/usr/bin/env python3
import argparse
import importlib
import os
from pathlib import Path
from typing import Iterable

import hydra
import torch
from omegaconf import OmegaConf
from torch import optim
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_cosine_schedule_with_warmup

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
from google.cloud import storage

from edmformer_gcs_dataset import Dataset as GcsDataset


def _is_gcs(path: str) -> bool:
    return path.startswith("gs://")


def _parse_gcs(uri: str) -> tuple[str, str]:
    bucket, blob = uri[5:].split("/", 1)
    return bucket, blob


def _read_lines(path: str) -> list[str]:
    if _is_gcs(path):
        bucket, blob = _parse_gcs(path)
        client = storage.Client()
        data = client.bucket(bucket).blob(blob).download_as_text()
        return data.splitlines()
    return Path(path).read_text(encoding="utf-8").splitlines()


def load_ids(path: str) -> list[str]:
    ids: list[str] = []
    for line in _read_lines(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line)
    return ids


def make_folds(ids: list[str], k: int, seed: int) -> list[list[str]]:
    rng = torch.Generator()
    rng.manual_seed(seed)
    perm = torch.randperm(len(ids), generator=rng).tolist()
    shuffled = [ids[i] for i in perm]
    folds = [[] for _ in range(k)]
    for i, item in enumerate(shuffled):
        folds[i % k].append(item)
    return folds


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


def evaluate(model, eval_loader, device):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in eval_loader:
            if batch is None:
                continue
            batch = {
                key: (val.to(device) if isinstance(val, torch.Tensor) else val)
                for key, val in batch.items()
            }
            res = model.infer_with_metrics(batch, prefix=None)
            results.append(res)
    if not results:
        return {}
    keys = results[0].keys()
    avg = {}
    for k in keys:
        avg[k] = float(sum(r[k] for r in results) / len(results))
    return avg


def build_dataset(hparams, base_abstracts: list[dict], split_ids: list[str]):
    abstracts = []
    for item in base_abstracts:
        new_item = dict(item)
        new_item["split_ids"] = split_ids
        abstracts.append(new_item)
    return GcsDataset(dataset_abstracts=abstracts, hparams=hparams)


def train_fold(index: int, args):
    device = xm.xla_device()
    torch.manual_seed(args.init_seed)

    hparams = OmegaConf.load(args.config)
    module = importlib.import_module("models." + args.model_name)
    Model = getattr(module, "Model")

    base_abstracts = OmegaConf.to_container(
        hparams.train_dataset.dataset_abstracts, resolve=True
    )
    if not base_abstracts:
        raise ValueError("No dataset_abstracts found in config.")

    split_source = base_abstracts[0]["split_ids_path"]
    all_ids = load_ids(split_source)

    test_ids = []
    if args.test_ids_path:
        test_ids = load_ids(args.test_ids_path)
        test_set = set(test_ids)
        before = len(all_ids)
        all_ids = [x for x in all_ids if x not in test_set]
        if xm.is_master_ordinal():
            xm.master_print(
                f"Excluded {before - len(all_ids)} test IDs from CV pool (test set size: {len(test_ids)})."
            )
        if len(all_ids) == 0:
            raise ValueError("No IDs left for CV after excluding test set.")

    if args.cv_folds < 2:
        folds = [all_ids]
    else:
        if len(all_ids) < args.cv_folds:
            raise ValueError(
                f"Not enough IDs for {args.cv_folds} folds after excluding test set."
            )
        folds = make_folds(all_ids, args.cv_folds, args.cv_seed)

    fold_metrics = []

    for fold_idx in range(max(1, args.cv_folds)):
        if args.cv_folds < 2:
            train_ids = all_ids
            val_ids = all_ids
        else:
            val_ids = folds[fold_idx]
            train_ids = [x for i, fold in enumerate(folds) if i != fold_idx for x in fold]

        model = Model(hparams).to(device)
        optimizer = optim.Adam(model.parameters(), **hparams.optimizer)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=hparams.warmup_steps,
            num_training_steps=hparams.total_steps,
        )

        train_dataset = build_dataset(hparams.train_dataset.hparams, base_abstracts, train_ids)
        eval_dataset = build_dataset(hparams.eval_dataset.hparams, base_abstracts, val_ids)

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
        train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)

        eval_loader = None
        if xm.is_master_ordinal():
            eval_loader = DataLoader(
                eval_dataset,
                **hparams.eval_dataloader,
                collate_fn=eval_dataset.collate_fn,
            )

        global_step = 0
        fold_checkpoint_dir = None
        if args.checkpoint_dir:
            fold_checkpoint_dir = os.path.join(args.checkpoint_dir, f"fold{fold_idx}")
            if xm.is_master_ordinal():
                global_step = load_checkpoint(
                    fold_checkpoint_dir, model, optimizer, scheduler
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
                        f"fold={fold_idx} epoch={epoch} step={global_step} "
                        f"loss={loss.item():.4f} loss_section={losses['loss_section'].item():.4f} "
                        f"loss_function={losses['loss_function'].item():.4f}"
                    )

                if fold_checkpoint_dir and global_step % args.save_interval == 0:
                    xm.rendezvous("save_ckpt")
                    if xm.is_master_ordinal():
                        save_checkpoint(
                            fold_checkpoint_dir, model, optimizer, scheduler, global_step
                        )
                    xm.rendezvous("save_ckpt_done")

            if global_step >= max_steps:
                break

        if xm.is_master_ordinal() and eval_loader is not None:
            metrics = evaluate(model, eval_loader, device)
            fold_metrics.append(metrics)
            xm.master_print(f"Fold {fold_idx} metrics: {metrics}")

    if xm.is_master_ordinal() and fold_metrics:
        keys = fold_metrics[0].keys()
        cv = {}
        for k in keys:
            cv[k] = float(sum(m.get(k, 0.0) for m in fold_metrics) / len(fold_metrics))
        xm.master_print(f"CV metrics (mean over {len(fold_metrics)} folds): {cv}")


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
    parser.add_argument("--cv_folds", type=int, default=1)
    parser.add_argument("--cv_seed", type=int, default=42)
    parser.add_argument(
        "--test-ids-path",
        type=str,
        default=os.environ.get("TEST_IDS_PATH_GCS"),
        help="Optional test IDs file (local or gs://). Excluded from CV folds.",
    )
    args = parser.parse_args()

    xmp.spawn(train_fold, args=(args,), nprocs=args.num_cores, start_method="fork")


if __name__ == "__main__":
    main()
