"""Prepare audio data for compressing speech SSL."""

from argparse import ArgumentParser
from pathlib import Path
from typing import Union
from tqdm import tqdm

import torchaudio


def create_tsv(
    root_dir: Union[str, Path],
    out_dir: Union[str, Path],
    extension: str = "flac",
) -> None:
    """Create file lists for training and validation.
    Args:
        root_dir (str or Path): The directory of the dataset.
        out_dir (str or Path): The directory to store the file lists.
        extension (str, optional): The extension of audio files. (Default: ``flac``)

    Returns:
        None
    """
    root_dir = Path(root_dir)
    out_dir = Path(out_dir)

    if not out_dir.exists():
        out_dir.mkdir()

    with open(
        out_dir / "train100.tsv", "w"
    ) as train100_f, open(
        out_dir / "train960.tsv", "w"
    ) as train960_f, open(
        out_dir / "valid.tsv", "w"
    ) as valid_f:
        print(root_dir, file=train100_f)
        print(root_dir, file=train960_f)
        print(root_dir, file=valid_f)

        for fname in tqdm(root_dir.glob(f"**/*.{extension}")):
            line = f"{fname.relative_to(root_dir)}\t{torchaudio.info(fname).num_frames}"

            if "train-clean-100" in str(fname):
                print(line, file=train100_f)
            if "train" in str(fname):
                print(line, file=train960_f)
            if "dev" in str(fname):
                print(line, file=valid_f)

    print("Finished creating the file lists successfully")


def parse_args():
    parser = ArgumentParser(
        description="Prepare audio data."
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the original dataset."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/librispeech"),
        help="Path to save the output."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    
    assert args.data.is_dir(), args.data
    args.out.mkdir(parents=True, exist_ok=True)

    create_tsv(
        root_dir=args.data,
        out_dir=args.out,
    )
