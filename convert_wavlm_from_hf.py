"""Convert Hugging Face's WavLM to our format."""

import torch
from transformers import WavLMModel

from wav2vec2.model import wav2vec2_model
from wav2vec2.utils.import_huggingface_wavlm import import_huggingface_model


if __name__ == "__main__":
    out_name = "pretrained/wavlm-base-plus.hf.pth"

    original = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
    imported = import_huggingface_model(original)
    imported.eval()
    print(imported)

    # default config of wavlm base
    wavlm_base_plus_config = dict(
        extractor_mode="group_norm",    # wavlm base only uses a group norm at the first conv layer
        extractor_conv_layer_config=[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2,
        extractor_conv_bias=False,
        encoder_embed_dim=768,
        encoder_projection_dropout=0.1,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=12,
        encoder_use_attention=[True] * 12,
        encoder_use_feed_forward=[True] * 12,
        encoder_total_num_heads=[12] * 12,
        encoder_remaining_heads=[list(range(12)) for _ in range(12)],
        encoder_num_buckets=320,
        encoder_max_distance=800,
        encoder_attention_dropout=0.1,
        encoder_ff_interm_features=[3072] * 12,
        encoder_ff_interm_dropout=0.0,
        encoder_dropout=0.1,
        encoder_layer_norm_first=False,     # wavlm base uses post norm
        encoder_layer_drop=0.05,
        aux_num_out=None,
        normalize_waveform=False,
        extractor_prune_conv_channels=False,
        encoder_prune_attention_heads=False,
        encoder_prune_attention_layer=False,
        encoder_prune_feed_forward_intermediate=False,
        encoder_prune_feed_forward_layer=False,
    )

    torch.save(
        {
            'state_dict': imported.state_dict(),
            'config': wavlm_base_plus_config,
        }, 
        out_name
    )

    # verify the saved ckpt
    ckpt = torch.load(out_name, map_location="cpu")
    model = wav2vec2_model(**ckpt['config'])
    print(model.load_state_dict(ckpt['state_dict'], strict=False))
