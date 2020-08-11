import click, logging, soundfile as sf
import pandas as pd, numpy as np
import noisereduce
from . import signal

def denoise(wav, fs, template_len, silence_ratio = 0.05):
    #wav = signal.wiener(wav, mysize=32)
    #wav = signal.wiener(wav, mysize=128)
    #wav = signal.bandpass(wav, min_freq, max_freq, fs)

    hop = int(template_len/2)
    energy_idx = pd.Series(wav).abs().rolling(template_len).sum()[::hop].sort_values().index.values - template_len
    noise_clips = [wav[energy_idx[i]:energy_idx[i]+template_len] for i in range(int(silence_ratio*len(energy_idx)))]
    print('Silence ratio {} --> {} noise clips'.format(silence_ratio, len(noise_clips)))
    if not noise_clips:
        return wav
    return noisereduce.reduce_noise(wav, noise_clips)

@click.command()
@click.argument('fname_in', type=click.Path(exists=True))
@click.argument('fname_out', type=click.Path())
@click.option('--template_len', type=click.INT, default=4096)
@click.option('--silence_ratio', type=click.FLOAT, default=0.05)
@click.option('--max_iterations', type=click.INT, default=5)
def main(fname_in, fname_out, template_len, silence_ratio, max_iterations):
    """ Denoise
    """
    logger = logging.getLogger(__file__)
    wav, fs = sf.read(fname_in)
    logger.info('Denoising '+fname_in)
    sf.write(fname_out, denoise(wav, fs, template_len, max_iterations, logger), fs)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()

