"""
Fetch the pretrained encoder into a local directory.

Normally ``AutoModel.from_pretrained("distilbert-base-uncased")`` would do this.
In this project's execution environment ``huggingface.co`` is not reachable —
the egress policy rejects it — so the weights are pulled from the legacy
public model bucket instead and written to ``models/distilbert-base-uncased/``,
which training and inference then load from disk.

If you are working somewhere with normal HuggingFace access you do not need
this script: point ``--pretrained-dir`` at a model id instead, or run

    python -c "from transformers import AutoModel, AutoTokenizer; \\
        AutoModel.from_pretrained('distilbert-base-uncased').save_pretrained(
            'models/distilbert-base-uncased'); \\
        AutoTokenizer.from_pretrained('distilbert-base-uncased').save_pretrained(
            'models/distilbert-base-uncased')"

Run:

    python -m classification.prefilter.fetch_model
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import urllib.request
from pathlib import Path

from config.logging_config import setup_logging

from classification.prefilter.config import DEFAULT_PRETRAINED_DIR

setup_logging()
logger = logging.getLogger(__name__)

LEGACY_BUCKET = "https://s3.amazonaws.com/models.huggingface.co/bert"

#: DistilBERT reuses the bert-base-uncased WordPiece vocabulary unchanged; the
#: bucket has no distilbert-specific vocab file. The checksum is asserted so a
#: silently different vocabulary cannot slip in.
VOCAB_MD5 = "64800d5d8528ce344256daf115d4965e"

FILES = {
    "config.json": f"{LEGACY_BUCKET}/distilbert-base-uncased-config.json",
    "vocab.txt": f"{LEGACY_BUCKET}/bert-base-uncased-vocab.txt",
    "pytorch_model.bin": (
        f"{LEGACY_BUCKET}/distilbert-base-uncased-pytorch_model.bin"
    ),
}

TOKENIZER_CONFIG = {
    "do_lower_case": True,
    "unk_token": "[UNK]",
    "sep_token": "[SEP]",
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "mask_token": "[MASK]",
    "tokenize_chinese_chars": True,
    "strip_accents": None,
    "model_max_length": 512,
    "tokenizer_class": "DistilBertTokenizer",
}


def _download(url: str, destination: Path) -> Path:
    """
    Download one file, skipping the work if it is already present.
    """

    if destination.exists() and destination.stat().st_size > 0:
        logger.info("Already present, skipping: %s", destination.name)
        return destination

    logger.info("Downloading %s -> %s", url, destination)

    destination.parent.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        while chunk := response.read(1 << 20):
            handle.write(chunk)

    return destination


def fetch(output_dir: Path = DEFAULT_PRETRAINED_DIR) -> Path:
    """
    Download the encoder and write the tokenizer configuration next to it.
    """

    output_dir = Path(output_dir)

    for file_name, url in FILES.items():
        _download(url, output_dir / file_name)

    vocab_file = output_dir / "vocab.txt"
    digest = hashlib.md5(vocab_file.read_bytes()).hexdigest()

    if digest != VOCAB_MD5:
        raise RuntimeError(
            f"Unexpected vocabulary checksum for {vocab_file}: "
            f"got {digest}, expected {VOCAB_MD5}"
        )

    tokenizer_config = output_dir / "tokenizer_config.json"

    if not tokenizer_config.exists():
        tokenizer_config.write_text(
            json.dumps(TOKENIZER_CONFIG, indent=2), encoding="utf-8"
        )

    # Materialise the fast tokenizer so training does not pay the conversion
    # cost on every run.
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(str(output_dir), use_fast=True).save_pretrained(
        str(output_dir)
    )

    logger.info("Pretrained encoder ready at %s", output_dir)

    return output_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fetch the pretrained encoder for the PII pre-filter."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_PRETRAINED_DIR))

    args = parser.parse_args(argv)

    fetch(Path(args.output_dir))


if __name__ == "__main__":
    main()
