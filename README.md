# Deep Neural Network-based Singing Synthesis

Contains all the code for my final year project, Deep Neural Network-based Singing Synthesis.

## Datasets

There are two datasets in `utils/datasets.py`, Choral Singing Dataset and VocalSet.
These have the base class `torch.utils.data.Dataset`.
When run for the first time they will download their datasets as needed.
VocalSet will also create a cache of each sample, ~10GB large.

## Models

There are 3 models:
- `models/wavenet.py`:  Contains the conditional WaveNet model
- `models/tacotron.py`: Contains the Tacotron 2 model
- `models/sodium.py`:   Contains the Non-Attentive Tacotron model

All 3 have default hyperparameters found in `hparams.py`.

## Training

To train the WaveNet model, run `train_wavenet.py`.

To train the Non-Attentive Tacotron model, run `train_sodium.py`.

## Inference

The `generate.py` script will do end-to-end inference.
The arguments are:
- `input`: Input MusicXML file with ARPABET lyrics on the second line
- `-t`, `--tacotron-model`: Tacotron model file
- `-w`, `--wavenet-model`: WaveNet model file
- `-o`, `--output`: Output file name without extension
- `-S`, `--singer`: Singer ID string
- `-T`, `--technique`: Technique ID string
- `-s`, `--spec-only`: Only produce spectrogram without audio generation
- `-p`, `--const-phoneme`: Constant phoneme if the MusicXML file doesn't have lyrics
- `-g`, `--griffinlim`: Use Griffin-Lim algorithm for inference
- `--temperature`: Temperature for WaveNet softmax


The script will generate 3 files, a `.png` of the spectrogram, a `.pt` of the spectrogram, and a `.wav` LPCM file of the produced audio.

Model files can be downloaded in the releases section of this repository, https://github.com/charlieharutaka/fyp/releases.
