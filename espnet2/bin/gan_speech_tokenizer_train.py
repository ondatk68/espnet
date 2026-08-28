#!/usr/bin/env python3
"""Command-line entry point for GAN-based speech-tokenizer training."""

from torch.multiprocessing.spawn import ProcessContext

from espnet2.tasks import abs_task
from espnet2.tasks.gan_speech_tokenizer import GANSpeechTokenizerTask


def get_parser():
    """Return the GAN speech-tokenizer training parser."""
    return GANSpeechTokenizerTask.get_parser()


def main(cmd=None):
    """Run GAN-based speech-tokenizer training."""
    # abs_task only imports ProcessContext when wandb is unavailable.  Supply
    # it locally so multi-GPU speech-tokenizer jobs also work with wandb
    # installed, without changing the shared task implementation.
    abs_task.ProcessContext = ProcessContext
    GANSpeechTokenizerTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
