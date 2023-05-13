from data import get_data, make_alphabet
from seq2seq import Encoder, Decoder, AttnDecoder
from learn import train, predict
from argparse import ArgumentParser
import torch, wandb, os
from pathlib import Path
import pandas as pd

parser = ArgumentParser()
parser.add_argument('-ct', '--cell_type', type=str.lower, default='lstm', choices=['rnn', 'gru', 'lstm'], help='Cell type for the encoder & decoder')
parser.add_argument('-a', '--attention', action='store_true', help='Whether to use attention mechanism in the decoder')
parser.add_argument('-es', '--embed_size', type=int, default=16, help='Embedding size for the alphabets')
parser.add_argument('-hs', '--hidden_size', type=int, default=256, help='Hidden size for the encoder & decoder')
parser.add_argument('-el', '--encoder_layers', type=int, default=1, help='Number of layers in the encoder')
parser.add_argument('-dl', '--decoder_layers', type=int, default=2, help='Number of layers in the decoder')
parser.add_argument('-do', '--dropout', type=float, default=0.3, help='Dropout rate for the encoder & decoder')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate for optimizing encoder & decoder')
parser.add_argument('-wd', '--weight_decay', type=float, default=0., help='Weight decay used for optimizing encoder & decoder')
parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train the model for')
parser.add_argument('-ea', '--early_stop', action='store_true', help='Whether to use early stopping')
parser.add_argument('-p', '--patience', type=int, default=1, help='Patience in epochs to use for early stopping. Ignored if --early_stop is not passed')
parser.add_argument('-d', '--device', type=str, default='cpu', help='Device to perform computations on')
parser.add_argument('-we', '--wandb_entity', type=str, default=None, help='WandB entity whose run is to be tracked. Locally logged in WandB entity would be used if not passed')
parser.add_argument('-wp', '--wandb_project', type=str, default=None, help='WandB project to log the run on. Logging is skipped if a value is not passed')
parser.add_argument('-run', '--run_test', action='store_true', help='Whether to run the trained model on test data. Predictions are stored in a folder named "test_preds"')
parser.add_argument('-s', '--save_model', action='store_true', help='Whether to save the trained model (in a directory named models)')

args = parser.parse_args()

device = 'cpu'
if args.device == 'cuda':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        print('[WARNING] CUDA device not available. Computing on CPU.')

train_data_given, train_data_target = get_data()
val_data_given, val_data_target = get_data(split='valid')
eng_alphabet, tam_alphabet = make_alphabet(train_data_given), make_alphabet(train_data_target)
cell_mapping = {
    'rnn': torch.nn.RNN,
    'gru': torch.nn.GRU,
    'lstm': torch.nn.LSTM
}
run_name = f'{"attn_" if args.attention else ""}{args.cell_type}_embed{args.embed_size}_hidden{args.hidden_size}_{args.encoder_layers}layerenc_{args.decoder_layers}layerdec_dropout{args.dropout}_wd{args.weight_decay}'

encoder = Encoder(eng_alphabet.letter_count, args.embed_size, args.hidden_size, cell_mapping[args.cell_type], args.encoder_layers, args.dropout).to(device, non_blocking=True)
if args.attention:
    decoder = AttnDecoder(args.hidden_size, args.embed_size, tam_alphabet.letter_count, cell_mapping[args.cell_type], args.decoder_layers, args.dropout).to(device, non_blocking=True)
else:
    decoder = Decoder(args.hidden_size, args.embed_size, tam_alphabet.letter_count, cell_mapping[args.cell_type], args.decoder_layers, args.dropout).to(device, non_blocking=True)
optimizer_enc = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
optimizer_dec = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
loss_fn = torch.nn.NLLLoss()

prev_score = 0.0
patience = args.patience

if args.wandb_project is not None:
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity)
    run.name = run_name

for epoch in range(args.epochs):
    train_loss, train_score = train(
        encoder=encoder,
        decoder=decoder,
        X=train_data_given,
        Y=train_data_target,
        alphabets=(eng_alphabet, tam_alphabet),
        optimizer_enc=optimizer_enc,
        optimizer_dec=optimizer_dec,
        device=device
    )
    val_preds, val_attns = predict(
        encoder=encoder,
        decoder=decoder,
        X=val_data_given,
        alphabets=(eng_alphabet, tam_alphabet),
        device=device
    )
    val_score = 0.0
    for val_pred, val_true in zip(val_preds, val_data_target):
        if val_pred == val_true+'EOW':
            val_score += 1.
    val_score /= len(val_data_target)

    print(f'Epoch {epoch+1}/{args.epochs} => Loss: {train_loss:.6f}, Accuracy: {train_score:.4f}, Validation accuracy: {val_score:.4f}')
    if args.wandb_project is not None:
        run.log({'epoch': epoch+1, 'loss': train_loss, 'accuracy': train_score, 'val_accuracy': val_score})

    if args.early_stop:
        if val_score > prev_score:
            patience = args.patience
        else:
            patience -= 1
            if patience == 0:
                break
        prev_score = val_score

if args.run_test:
    test_data_given, test_data_target = get_data(split='test')
    test_preds, test_attns = predict(
        encoder=encoder,
        decoder=decoder,
        X=test_data_given,
        alphabets=(eng_alphabet, tam_alphabet),
        device=device
    )
    test_score = 0.0
    for test_pred, test_true in zip(test_preds, test_data_target):
        if test_pred == test_true+'EOW':
            test_score += 1.
    test_score /= len(test_data_target)
    print(f'Test accuracy: {test_score:.4f}')
    if args.wandb_project is not None:
        run.log({'test_accuracy': test_score})
    preds_dir = Path('test_preds/')
    if not preds_dir.is_dir():
        os.mkdir(preds_dir)
    test_preds_wo_eow = [
        test_pred[:-3] if test_pred[-3:] == 'EOW' else test_pred for test_pred in test_preds
    ]
    test_preds_df = pd.DataFrame(data={
        'source': test_data_given,
        'truth': test_data_target,
        'prediction': test_preds_wo_eow
    })
    test_preds_df.to_csv(f'test_preds/{run_name}.csv', index=None)

if args.wandb_project is not None:
    run.finish()

if args.save_model:
    save_path = Path('models/')
    if not save_path.is_dir():
        os.mkdir(save_path)
    model_save_dir = f'models/{run_name}/'
    os.mkdir(model_save_dir)
    torch.save(encoder.state_dict(), f=model_save_dir+'encoder.pth')
    torch.save(decoder.state_dict(), f=model_save_dir+'decoder.pth')
