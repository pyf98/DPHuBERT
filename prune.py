import json
import pathlib
import torch
from argparse import ArgumentParser

from wav2vec2.model import (
    wav2vec2_model,
)


def prune_from_ckpt(distilled_ckpt, original_ckpt):
    ckpt = torch.load(distilled_ckpt, map_location='cpu')
    student_model_state_dict = {
        k[len("student_model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("student_model.")
    }
    distill_linear_projs_state_dict = {
        k[len("distill_linear_projs."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("distill_linear_projs.")
    }
    config = torch.load(original_ckpt, map_location='cpu')['config']
    config.update(
        dict(
            extractor_prune_conv_channels="feature_extractor.conv_layers.0.hard_concrete.log_alpha" in student_model_state_dict,
            encoder_prune_attention_heads="encoder.transformer.layers.0.attention.hard_concrete_for_heads.log_alpha" in student_model_state_dict,
            encoder_prune_attention_layer="encoder.transformer.layers.0.attention.hard_concrete_for_layer.log_alpha" in student_model_state_dict,
            encoder_prune_feed_forward_intermediate="encoder.transformer.layers.0.feed_forward.hard_concrete_for_intermediate.log_alpha" in student_model_state_dict,
            encoder_prune_feed_forward_layer="encoder.transformer.layers.0.feed_forward.hard_concrete_for_layer.log_alpha" in student_model_state_dict,
        )
    )
    model = wav2vec2_model(**config)
    model.load_state_dict(student_model_state_dict, strict=True)

    conv_config, use_attention, use_feed_forward, num_heads, remaining_heads, ff_interm_features = model.prune()
    pruned_config = config.copy()
    if len(num_heads) == 0:     # for wavlm
        assert len(remaining_heads) > 0
        pruned_config.update(
            {
                "encoder_remaining_heads": remaining_heads,
            }
        )
    else:
        pruned_config.update(
            {
                "encoder_num_heads": num_heads,
            }
        )
    pruned_config.update(
        {
            "extractor_conv_layer_config": conv_config,
            "encoder_use_attention": use_attention,
            "encoder_use_feed_forward": use_feed_forward,
            "encoder_ff_interm_features": ff_interm_features,
            "extractor_prune_conv_channels": False,
            "encoder_prune_attention_heads": False,
            "encoder_prune_attention_layer": False,
            "encoder_prune_feed_forward_intermediate": False,
            "encoder_prune_feed_forward_layer": False,
        }
    )
    print(json.dumps(pruned_config, indent=4))

    ret = {
        "state_dict": model.state_dict(),
        "config": pruned_config,
        "distill_linear_projs": distill_linear_projs_state_dict,
    }
    return ret


def load_pruned_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model = wav2vec2_model(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model


def parse_args():
    parser = ArgumentParser(description="Prune and save distilled model.")
    parser.add_argument(
        "--distilled_ckpt",
        type=pathlib.Path,
        help="Path to the distilled model checkpoint."
    )
    parser.add_argument(
        "--original_ckpt",
        type=pathlib.Path,
        help="Path to the original checkpoint."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    out_path = args.distilled_ckpt.parent / "pruned_hubert_base.pth"
    torch.save(
        prune_from_ckpt(
            distilled_ckpt=args.distilled_ckpt,
            original_ckpt=args.original_ckpt
        ),
        out_path
    )

    # Check if loading from ckpt works
    load_pruned_model(out_path)

    print(f"Successfully saved pruned model weights and config to: {out_path}")
