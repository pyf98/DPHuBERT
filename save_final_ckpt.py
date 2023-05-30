from argparse import ArgumentParser
import json
import pathlib
import torch

from prune import load_pruned_model


def parse_args():
    parser = ArgumentParser(description="Save ckpt and config after final distill.")
    parser.add_argument(
        "--config_path",
        type=pathlib.Path,
        help="Path to the checkpoint file containing the pruned config."
    )
    parser.add_argument(
        "--ckpt_after_final_distill",
        type=pathlib.Path,
        help="Path to the checkpoint file after final distill."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    config = torch.load(args.config_path, map_location="cpu")["config"]
    print(json.dumps(config, indent=4))

    ckpt = torch.load(args.ckpt_after_final_distill, map_location="cpu")
    student_model_state_dict = {
        k[len("student_model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("student_model.")
    }
    distill_linear_projs_state_dict = {
        k[len("distill_linear_projs."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("distill_linear_projs.")
    }

    out_path = args.ckpt_after_final_distill.parent / "pruned_hubert_base.pth"
    torch.save(
        {
            "state_dict": student_model_state_dict,
            "config": config,
            "distill_linear_projs": distill_linear_projs_state_dict,
        },
        out_path
    )
    
    load_pruned_model(out_path)     # verify if it works
    print(f"Successfully saved pruned model weights and config to: {out_path}")
