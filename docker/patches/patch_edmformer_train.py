#!/usr/bin/env python3
from pathlib import Path


def main() -> None:
    path = Path("/app/third_party/EDMFormer/src/SongFormer/train/train.py")
    text = path.read_text()
    import re

    # Patch 1: ensure model_ema exists on all ranks (avoids UnboundLocalError).
    if "model_ema = None" not in text:
        pattern = r"\n(\s*params = model\.parameters\(\))\n"
        match = re.search(pattern, text)
        if not match:
            raise SystemExit(
                "Patch failed: expected 'params = model.parameters()' line not found in train.py. "
                "The upstream file may have changed."
            )

        indent = match.group(1).split("params =")[0]
        replacement = f"\n{match.group(1)}\n{indent}model_ema = None"
        text = re.sub(pattern, replacement, text, count=1)

    # Patch 2: enable find_unused_parameters via Accelerate DDP kwargs.
    if "DistributedDataParallelKwargs" not in text:
        import_pattern = r"from accelerate\.utils import ([^\n]+)"
        import_match = re.search(import_pattern, text)
        if not import_match:
            raise SystemExit(
                "Patch failed: expected accelerate.utils import line not found in train.py. "
                "The upstream file may have changed."
            )

        imports = import_match.group(1).strip()
        if "DistributedDataParallelKwargs" not in imports:
            new_imports = imports + ", DistributedDataParallelKwargs"
            text = re.sub(import_pattern, f"from accelerate.utils import {new_imports}", text, count=1)

    # Inject kwargs_handlers into the Accelerator(...) call if missing.
    lines = text.splitlines()
    accel_idx = None
    for i, line in enumerate(lines):
        if "accelerator = Accelerator(" in line:
            accel_idx = i
            break

    if accel_idx is None:
        raise SystemExit(
            "Patch failed: expected 'accelerator = Accelerator(' block not found in train.py. "
            "The upstream file may have changed."
        )

    close_idx = None
    for j in range(accel_idx + 1, len(lines)):
        if lines[j].strip() == ")":
            close_idx = j
            break

    if close_idx is None:
        raise SystemExit(
            "Patch failed: could not find end of Accelerator(...) call in train.py. "
            "The upstream file may have changed."
        )

    accelerator_block = "\n".join(lines[accel_idx : close_idx + 1])
    if "kwargs_handlers" not in accelerator_block:
        indent = lines[accel_idx].split("accelerator = Accelerator(")[0]
        arg_indent = indent + "    "
        lines.insert(
            close_idx,
            f"{arg_indent}kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],",
        )
        text = "\n".join(lines) + "\n"

    # Patch 3: avoid Balancer autograd.grad with DDP (causes unused-parameter errors).
    if "use_balancer = accelerator.num_processes == 1" not in text:
        lines = text.splitlines()
        bal_idx = None
        for i, line in enumerate(lines):
            if "balancer = Balancer(" in line:
                bal_idx = i
                break
        if bal_idx is None:
            raise SystemExit(
                "Patch failed: expected 'balancer = Balancer(' block not found in train.py. "
                "The upstream file may have changed."
            )
        close_idx = None
        for j in range(bal_idx + 1, len(lines)):
            if lines[j].strip() == ")":
                close_idx = j
                break
        if close_idx is None:
            raise SystemExit(
                "Patch failed: could not find end of Balancer(...) block in train.py. "
                "The upstream file may have changed."
            )
        indent = lines[bal_idx].split("balancer =")[0]
        lines.insert(close_idx + 1, f"{indent}use_balancer = accelerator.num_processes == 1")
        text = "\n".join(lines) + "\n"

    # Replace loss_sum computation to bypass Balancer on multi-GPU.
    if "if use_balancer:" not in text:
        if "loss_sum = losses[\"loss_section\"] + losses[\"loss_function\"]" not in text:
            lines = text.splitlines()
            start_idx = None
            for i, line in enumerate(lines):
                if "loss_sum = balancer.cal_mix_loss(" in line:
                    start_idx = i
                    break
            if start_idx is not None:
                indent = lines[start_idx].split("loss_sum =")[0]
                # Track parentheses to find the end of the call.
                paren_count = lines[start_idx].count("(") - lines[start_idx].count(")")
                end_idx = start_idx
                for j in range(start_idx + 1, len(lines)):
                    paren_count += lines[j].count("(") - lines[j].count(")")
                    if paren_count == 0:
                        end_idx = j
                        break
                if end_idx == start_idx:
                    raise SystemExit(
                        "Patch failed: could not find end of balancer.cal_mix_loss(...) call. "
                        "The upstream file may have changed."
                    )

                block = lines[start_idx : end_idx + 1]
                extra_indent = indent + "    "
                wrapped = [f"{indent}if use_balancer:"]
                for line in block:
                    wrapped.append(extra_indent + line[len(indent) :])
                wrapped.append(f"{indent}else:")
                wrapped.append(
                    f"{indent}    loss_sum = losses[\"loss_section\"] + losses[\"loss_function\"]"
                )
                lines[start_idx : end_idx + 1] = wrapped
                text = "\n".join(lines) + "\n"
            else:
                print(
                    "Patch warning: loss_sum = balancer.cal_mix_loss(...) block not found; "
                    "skipping Balancer patch."
                )

    if "params = model.parameters()" not in text:
        raise SystemExit(
            "Patch failed: expected 'params = model.parameters()' line not found after patching. "
            "The upstream file may have changed."
        )

    # Patch 4: ensure all ranks skip batches consistently in the training loop only.
    if "skip_batch = torch.tensor(1, device=device)" not in text:
        lines = text.splitlines()
        target_idx = None
        for i, line in enumerate(lines):
            if "for step, batch in enumerate(data_loader):" in line:
                # Find the first `if batch is None:` after the training loop starts.
                for j in range(i + 1, min(i + 120, len(lines))):
                    if lines[j].strip() == "if batch is None:":
                        target_idx = j
                        break
                break
        if target_idx is None:
            raise SystemExit(
                "Patch failed: expected training-loop 'if batch is None:' line not found in train.py. "
                "The upstream file may have changed."
            )
        if target_idx + 1 >= len(lines) or lines[target_idx + 1].strip() != "continue":
            raise SystemExit(
                "Patch failed: expected 'continue' after 'if batch is None:' in training loop. "
                "The upstream file may have changed."
            )

        indent = lines[target_idx].split("if batch is None:")[0]
        replacement = [
            f"{indent}if batch is None:",
            f"{indent}    skip_batch = torch.tensor(1, device=device)",
            f"{indent}else:",
            f"{indent}    skip_batch = torch.tensor(0, device=device)",
            f"{indent}if accelerator.gather(skip_batch).max().item() > 0:",
            f"{indent}    continue",
        ]
        lines[target_idx : target_idx + 2] = replacement
        text = "\n".join(lines) + "\n"

    # Patch 5: synchronize early stopping across ranks to avoid collective mismatch.
    if "stop_flag = accelerator.gather(should_stop).max().item()" not in text:
        lines = text.splitlines()
        # Insert should_stop initializer after eval wait_for_everyone inside eval block.
        inserted_init = False
        for i, line in enumerate(lines):
            if line.strip() == "accelerator.wait_for_everyone()":
                # Heuristic: choose the eval block one with extra indentation (inside training loop).
                if line.startswith("                            ") and i + 1 < len(lines):
                    indent = line.split("accelerator.wait_for_everyone()")[0]
                    lines.insert(
                        i + 1,
                        f"{indent}should_stop = torch.tensor(0, device=device)",
                    )
                    inserted_init = True
                    break
        if not inserted_init:
            raise SystemExit(
                "Patch failed: could not insert should_stop initializer after eval wait_for_everyone()."
            )

        # Replace early stopping break with should_stop flag set.
        replaced = False
        for i in range(len(lines) - 2):
            if (
                lines[i].strip() == "if no_improve_steps >= early_stop_patience:"
                and lines[i + 1].strip() == "print(\"Early stopping triggered.\")"
                and lines[i + 2].strip() == "break"
            ):
                indent = lines[i].split("if no_improve_steps >= early_stop_patience:")[0]
                lines[i : i + 3] = [
                    f"{indent}if no_improve_steps >= early_stop_patience:",
                    f"{indent}    print(\"Early stopping triggered.\")",
                    f"{indent}    should_stop = torch.tensor(1, device=device)",
                ]
                replaced = True
                break
        if not replaced:
            raise SystemExit(
                "Patch failed: expected early stopping break block not found."
            )

        # Insert stop_flag sync before the outer wait_for_everyone() after eval block.
        inserted_stop = False
        for i, line in enumerate(lines):
            if line.strip() == "accelerator.wait_for_everyone()":
                if line.startswith("                        "):
                    indent = line.split("accelerator.wait_for_everyone()")[0]
                    block = [
                        f"{indent}if accelerator.sync_gradients and global_step % args.eval_interval == 0:",
                        f"{indent}    stop_flag = accelerator.gather(should_stop).max().item()",
                        f"{indent}    if stop_flag > 0:",
                        f"{indent}        break",
                    ]
                    lines[i:i] = block
                    inserted_stop = True
                    break
        if not inserted_stop:
            raise SystemExit(
                "Patch failed: could not insert stop_flag sync before wait_for_everyone()."
            )

        text = "\n".join(lines) + "\n"

    path.write_text(text)
    print("Patch applied.")

    # Patch SongFormer loss initialization to keep graph for DDP.
    model_path = Path("/app/third_party/EDMFormer/src/SongFormer/models/SongFormer.py")
    model_text = model_path.read_text()
    if "loss = 0.0" in model_text:
        model_text = model_text.replace(
            "loss = 0.0",
            "loss = torch.zeros((), device=outputs[\"function_logits\"].device)",
            1,
        )
        model_path.write_text(model_text)
        print("Patched SongFormer compute_losses loss init.")


if __name__ == "__main__":
    main()
