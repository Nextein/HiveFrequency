# ShaBoom DnB Transformer

Custom Transformer trained on Drum and Bass audio to generate audio samples as output.

The model uses VQ-VAE to encode and decode spectrograms into discrete token representations. The autoregressive model then learns the distibution of these tokens and generates samples from it. These samples are then decoded back into raw audio.

This approach speeds up training of the transformer by learning more efficient representations of the data in a latent space of smaller dimensions.

The use of an open-source encoder/decoder pre-trained on music will greatly improve performance and speed up the training process compared to training from scratch.

## Folder structure

    data/
        raw_audio/
        spectrograms
        discrete_tokens/
        output_tokens/
        output_audio/

