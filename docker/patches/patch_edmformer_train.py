#!/usr/bin/env python3
from pathlib import Path


def main() -> None:
    path = Path("/app/third_party/EDMFormer/src/SongFormer/train/train.py")
    text = path.read_text()
    if "model_ema = None" in text:
        print("Patch already applied.")
        return

    needle = "    params = model.parameters()\\n\\n    if accelerator.is_main_process:"
    replacement = (
        "    params = model.parameters()\\n\\n"
        "    model_ema = None\\n"
        "    if accelerator.is_main_process:"
    )
    if needle not in text:
        raise SystemExit(
            "Patch failed: expected block not found in train.py. "
            "The upstream file may have changed."
        )

    path.write_text(text.replace(needle, replacement))
    print("Patch applied.")


if __name__ == "__main__":
    main()
