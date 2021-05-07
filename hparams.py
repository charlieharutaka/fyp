from models.tacotron import DecoderCell

TACOTRON_HP = {
    "encoder_lyric_dim": 256,
    "encoder_pitch_dim": 256,
    "encoder_rhythm_dim": 256,
    "embedding_dim": 256,
    "encoder_n_convolutions": 3,
    "encoder_kernel_size": 5,
    "encoder_p_dropout": 0.5,
    "hidden_dim": 256,
    "attention_dim": 128,
    "attention_rnn_dim": 1024,
    "attention_location_n_filters": 32,
    "attention_location_kernel_size": 31,
    "decoder_rnn_dim": 1024,
    "p_prenet_dropout": 0.1,
    "p_attention_dropout": 0.1,
    "p_decoder_dropout": 0.1,
    "prenet_dim": 128,
    "max_decoder_steps": 1000,
    "stopping_threshold": 0.5,
    "postnet_n_convolutions": 5,
    "postnet_embedding_dim": 512,
    "postnet_kernel_size": 5,
    "postnet_p_dropout": 0.5,
    "decoder_cell": DecoderCell
}

WAVENET_HP = {
    "layers": 10,
    "blocks": 4,
    "in_channels": 1,
    "cond_in_channels": 64,
    "cond_channels": 32,
    "dilation_channels": 32,
    "residual_channels": 32,
    "skip_channels": 256,
    "end_channels": 256,
    "classes": 256,
    "kernel_size": 2,
    "bias": False
}
