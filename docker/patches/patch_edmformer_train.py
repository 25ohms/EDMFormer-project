#!/usr/bin/env python3
from pathlib import Path


def main() -> None:
    path = Path("/app/third_party/EDMFormer/src/SongFormer/train/train.py")
    text = path.read_text()
    import re

    # Patch 1: ensure model_ema exists on all ranks (avoids UnboundLocalError).
    if "model_ema = None" not in text:
        pattern = r"(\n\s*params = model\.parameters\(\)\n)(\s*if accelerator\.is_main_process:)"
        match = re.search(pattern, text)
        if not match:
            raise SystemExit(
                "Patch failed: expected 'params = model.parameters()' block not found in train.py. "
                "The upstream file may have changed."
            )

        replacement = match.group(1) + "    model_ema = None\\n" + match.group(2)
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
    if "kwargs_handlers" not in text:
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

        # Find the closing parenthesis of the Accelerator(...) call.
        indent = lines[accel_idx].split("accelerator = Accelerator(")[0]
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

        arg_indent = indent + "    "
        lines.insert(
            close_idx,
            f"{arg_indent}kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],",
        )
        text = "\n".join(lines) + "\n"

    pattern = r"(\n\s*params = model\.parameters\(\)\n)(\s*if accelerator\.is_main_process:)"
    if not re.search(pattern, text):
        raise SystemExit(
            "Patch failed: expected 'params = model.parameters()' block not found after patching. "
            "The upstream file may have changed."
        )

    path.write_text(text)
    print("Patch applied.")


if __name__ == "__main__":
    main()
