# Speech Tokenizer

This directory is the ESPnet2 recipe template for training a differentiable
speech tokenizer with ASR and speaker-conditioned reconstruction objectives.

## Recipe flow

1. Prepare Kaldi-style data directories.
2. Apply optional speed perturbation.
3. Format and resample waveforms.
4. Filter training and validation utterances by duration.
5. Prepare the ASR token list.
6. Extract one speaker embedding per utterance.
7. Train offline K-means centroids, or validate an external centroid model.
8. Collect shapes for speech, text, and speaker embeddings.
9. Train the speech tokenizer and its downstream objectives.
10. Extract discrete token sequences without loading downstream objectives.

The adversarial task uses two optimizers and
`SpeechTokenizerGANTrainer`. The non-adversarial task is selected with
`--speech_tokenizer_task speech_tokenizer` and will use one optimizer and the
standard trainer once its task and reconstruction objectives are implemented.
Speaker conditioning is required by both task variants.

## Creating a corpus recipe

From the ESPnet repository root, run:

```bash
egs2/TEMPLATE/speech_tokenizer1/setup.sh egs2/<corpus>/speech_tokenizer1
```

Then provide `local/data.sh`, a training configuration under `conf/`, and a
small `run.sh` that calls `speech_tokenizer.sh` with corpus-specific dataset
names. Common scripts and utilities are shared with the ASR template.

## K-means initialization

By default, stage 7 extracts `wavlm_large/21` features from ten percent of the
training data and learns 2000 centroids. To use an existing ESPnet ASR2 K-means
model instead, specify:

```bash
./speech_tokenizer.sh --centroid_path /path/to/km_2000.mdl ...
```

## Task selection

GAN-based ASR and reconstruction training is the current implementation:

```bash
./speech_tokenizer.sh \
    --speech_tokenizer_task gan_speech_tokenizer \
    --speech_tokenizer_config conf/train_gan.yaml \
    ...
```

Tokenizer-only inference writes `token` and `token_length` files. It does not
construct the ASR or reconstruction models and does not require speaker
embeddings at inference time.
