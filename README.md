# CS6910-A3
Programming assignment 3 done as part of CS6910 - Fundamentals of Deep Learning

## Goal

To build a Seq2Seq model for text transliteration from English to Indian languages, using the [Aksharantar dataset](https://drive.google.com/file/d/1uRKU4as2NlS9i8sdLRS1e326vQRdhvfw/view?usp=share_link) released by [AI4Bharat](https://ai4bharat.org/).

For running all code, the zipped data is expected to be downloaded and extracted to a subdirectory named `data`. Code may be changed to get the data from other paths as well.

**Language chosen: Tamil**

## Directory contents

- [data.py](./data.py): Functions to extract data and make an alphabet class
- [seq2seq.py](./seq2seq.py): Class definitions for encoder and decoders
- [learn.py](./learn.py): Train and predict functions for the seq2seq model
- [train.py](./train.py): Main abstract script, using the above helper scripts to build, train and evaluate a seq2seq model
- [sweep.py](./sweep.py): Helper script to run sweeps on WandB
- [A3.ipynb](./A3.ipynb): Notebook for implementation checking & analysis of results
- [models](./models/): Saved state_dicts of the best models, both with and without attention
- [predictions_vanilla](./predictions_vanilla/), [predictions_attention](./predictions_attention/): Saved predictions by the best models, with and without attention

## Usage for train.py

The following command line arguments are supported:

| Argument flag | Description | Default |
| :-: | :-: | :-: |
| `-ct`, `--cell_type` | Cell type for encoder and decoder | lstm |
| `-a`, `--attention` | Uses attention mechanism in the decoder if passed | N/A |
| `-es`, `--embed_size` | Embedding size for encoder and decoder | 16 |
| `-hs`, `--hidden_size` | Hidden state size for encoder and decoder | 256 |
| `-el`, `--encoder_layers` | Number of layers in the encoder | 1 |
| `-dl`, `--decoder_layers` | Number of layers in the decoder | 2 |
| `-do`, `--dropout` | Dropout rate for encoder and decoder | 0.3 |
| `-lr`, `--learning_rate` | Learning rate for optimizing encoder and decoder | ${10}^{-3}$ |
| `-wd`, `--weight_decay` | Weight decay for optimizing encoder and decoder | 0.0 |
| `-e`, `--epochs` | Number of epochs to train for | 10 |
| `-ea`, `--early_stop` | Trains with early stopping if passed | N/A |
| `-p`, `--patience` | Patience in epochs for early stopping | 1 |
| `-d`, `--device` | Device to use for computation | cpu |
| `-we`, `--wandb_entity` | WandB entity tracking the run | None |
| `-wp`, `--wandb_project` | WandB project to track the run on | None |
| `-run`, `--run_test` | Evaluates model on test data if passed | N/A |
| `-s`, `--save_model` | Saves the trained encoder and decoder if passed | N/A |

For more detailed usage information, run <br> 
``` bash
python3 train.py -h
```