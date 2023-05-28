"""Convert fairseq's HuBERT Large to our format."""

import torch
import fairseq
from torchaudio.models.wav2vec2.utils import import_fairseq_model

from wav2vec2.model import wav2vec2_model


if __name__ == "__main__":
    out_name = "pretrained/hubert-large-ll60k.fairseq.pth"

    fairseq_ckpt = "pretrained/fairseq/hubert_large_ll60k.pt"
    ensemble, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([fairseq_ckpt])
    original = ensemble[0]
    imported = import_fairseq_model(original)
    print(imported)
    
    # default config of hubert large
    hubert_large_config = dict(
        extractor_mode="layer_norm",
        extractor_conv_layer_config=[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2,
        extractor_conv_bias=False,
        encoder_embed_dim=1024,
        encoder_projection_dropout=0.1,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=24,
        encoder_use_attention=[True] * 24,
        encoder_use_feed_forward=[True] * 24,
        encoder_num_heads=[16] * 24,
        encoder_head_dim=64,
        encoder_attention_dropout=0.1,
        encoder_ff_interm_features=[4096] * 24,
        encoder_ff_interm_dropout=0.0,
        encoder_dropout=0.1,
        encoder_layer_norm_first=True,     # hubert large uses pre norm
        encoder_layer_drop=0.05,
        aux_num_out=None,
        normalize_waveform=True,
        extractor_prune_conv_channels=False,
        encoder_prune_attention_heads=False,
        encoder_prune_attention_layer=False,
        encoder_prune_feed_forward_intermediate=False,
        encoder_prune_feed_forward_layer=False,
    )

    torch.save(
        {
            'state_dict': imported.state_dict(),
            'config': hubert_large_config,
        }, 
        out_name
    )

    # verify the saved ckpt
    ckpt = torch.load(out_name, map_location="cpu")
    model = wav2vec2_model(**ckpt['config'])
    res = model.load_state_dict(ckpt['state_dict'], strict=False)
    print(f"Missing: {res.missing_keys}\nUnexpected: {res.unexpected_keys}")
