#!/usr/bin/env python3
from pathlib import Path


def main() -> None:
    path = Path("/app/third_party/EDMFormer/src/SongFormer/train/train.py")
    text = path.read_text()
    if "model_ema = None" in text:
        print("Patch already applied.")
        return

    import re

    pattern = r"(\n\s*params = model\.parameters\(\)\n)(\s*if accelerator\.is_main_process:)"
    match = re.search(pattern, text)
    if not match:
        raise SystemExit(
            "Patch failed: expected 'params = model.parameters()' block not found in train.py. "
            "The upstream file may have changed."
        )

    replacement = match.group(1) + "    model_ema = None\\n" + match.group(2)
    new_text = re.sub(pattern, replacement, text, count=1)
    path.write_text(new_text)
    print("Patch applied.")


if __name__ == "__main__":
    main()
