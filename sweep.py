from data import get_data, make_alphabet, word_to_tensor
from seq2seq import Encoder, Decoder, AttnDecoder
import torch
import wandb

train_data_given, train_data_target = get_data()
val_data_given, val_data_target = get_data(split='valid')
# train_data_given, train_data_target = train_data_given[:1000], train_data_target[:1000]
# val_data_given, val_data_target = val_data_given[:1000], val_data_target[:1000]
eng_alphabet = make_alphabet(train_data_given)
tam_alphabet = make_alphabet(train_data_target)
TFR = 0.5
MAX_LENGTH = 100

def perform_run(config=None):
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Computing on {device}')
    # device = 'cpu'

    with wandb.init(config=config) as run:
        config = wandb.config
        run.name = f'{config.cell_type}_embed{config.embedding_size}_hidden{config.hidden_size}_{config.encoder_layers}layerenc_{config.decoder_layers}layerdec_dropout{config.dropout}_wd{config.weight_decay}'
        if config.cell_type == 'lstm':
            cell_type_class = torch.nn.LSTM
        elif config.cell_type == 'gru':
            cell_type_class = torch.nn.GRU
        else:
            cell_type_class = torch.nn.RNN
        encoder = Encoder(eng_alphabet.letter_count, config.embedding_size, config.hidden_size, cell_type_class, config.encoder_layers, config.dropout if config.encoder_layers > 1 else 0.0).to(device, non_blocking=True)
        # decoder = Decoder(config.hidden_size, config.embedding_size, tam_alphabet.letter_count, cell_type_class, config.decoder_layers, config.dropout if config.decoder_layers > 1 else 0.0).to(device, non_blocking=True)
        decoder = AttnDecoder(config.hidden_size, config.embedding_size, tam_alphabet.letter_count, cell_type_class, config.decoder_layers, config.dropout if config.decoder_layers > 1 else 0.0).to(device, non_blocking=True)
        optimizer_enc = torch.optim.Adam(encoder.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        optimizer_dec = torch.optim.Adam(decoder.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        loss_fn = torch.nn.NLLLoss()

        prev_score = 0.0
        for epoch in range(config.epochs):
            train_loss = 0.0
            train_score, val_score = 0.0, 0.0

            for train_index in range(len(train_data_given)):
                hidden_enc = encoder.initHidden().to(device, non_blocking=True)
                cell_enc = encoder.initHidden().to(device, non_blocking=True)

                optimizer_enc.zero_grad()
                optimizer_dec.zero_grad()

                input_vector = word_to_tensor(eng_alphabet, train_data_given[train_index]).to(device, non_blocking=True)
                input_len = len(input_vector)

                outputs_enc = torch.zeros(MAX_LENGTH, encoder.hidden_size).to(device, non_blocking=True)

                loss = 0.0

                for i, char in enumerate(input_vector):
                    if config.cell_type == 'lstm':
                        output_enc, (hidden_enc, cell_enc) = encoder(char, (hidden_enc, cell_enc))
                    else:
                        output_enc, hidden_enc = encoder(char, hidden_enc)
                    outputs_enc[i] = outputs_enc[0, 0]
                
                input_dec = torch.tensor([[0]]).to(device, non_blocking=True)
                target_vector = word_to_tensor(tam_alphabet, train_data_target[train_index]).to(device, non_blocking=True)
                target_len = len(target_vector)

                # hidden_dec = hidden_enc.to(device, non_blocking=True)
                hidden_dec = torch.cat([hidden_enc[-1].reshape(1, 1, -1)]*decoder.num_layers).to(device, non_blocking=True)
                if config.cell_type == 'lstm':
                    cell_dec = torch.cat([cell_enc[-1].reshape(1, 1, -1)]*decoder.num_layers).to(device, non_blocking=True)

                use_teacher_forcing = True if torch.rand(1) < TFR else False
                prediction = ''

                if use_teacher_forcing:
                    for di in range(target_len):
                        if config.cell_type == 'lstm':
                            output_dec, (hidden_dec, cell_dec), attn = decoder(input_dec, (hidden_dec, cell_dec), outputs_enc)
                        else:
                            output_dec, hidden_dec, attn = decoder(input_dec, hidden_dec, outputs_enc)
                        loss += loss_fn(output_dec, target_vector[di])
                        input_dec = target_vector[di]
                        prediction += tam_alphabet.index_to_letter[output_dec.argmax(dim=1).squeeze().detach()]
                else:
                    for di in range(target_len):
                        if config.cell_type == 'lstm':
                            output_dec, (hidden_dec, cell_dec), attn = decoder(input_dec, (hidden_dec, cell_dec), outputs_enc)
                        else:
                            output_dec, hidden_dec, attn = decoder(input_dec, hidden_dec, outputs_enc)
                        input_dec = output_dec.argmax(dim=1).squeeze().detach()
                        loss += loss_fn(output_dec, target_vector[di])
                        prediction += tam_alphabet.index_to_letter[input_dec]
                        if input_dec.item() == 1:
                            break                

                loss.backward()
                optimizer_enc.step()
                optimizer_dec.step()

                train_loss += loss.detach() / input_len
                if prediction == train_data_target[train_index]+'EOW':
                    train_score += 1.
            
            train_loss /= len(train_data_given)
            train_score /= len(train_data_target)

            with torch.inference_mode(): 
                for val_index in range(len(val_data_given)):
                    input_vector = word_to_tensor(eng_alphabet, val_data_given[val_index]).to(device, non_blocking=True)
                    input_len = len(input_vector)
                    hidden_enc = encoder.initHidden().to(device, non_blocking=True)
                    cell_enc = encoder.initHidden().to(device, non_blocking=True)
                    outputs_enc = torch.zeros(MAX_LENGTH, encoder.hidden_size).to(device, non_blocking=True)

                    for i, char in enumerate(input_vector):
                        if config.cell_type == 'lstm':
                            output_enc, (hidden_enc, cell_enc) = encoder(char, (hidden_enc, cell_enc))
                        else:
                            output_enc, hidden_enc = encoder(char, hidden_enc)
                        outputs_enc[i] = output_enc[0, 0]

                    input_dec = torch.tensor([[0]]).to(device, non_blocking=True)
                    target_vector = word_to_tensor(tam_alphabet, val_data_target[val_index]).to(device, non_blocking=True)
                    hidden_dec = torch.cat([hidden_enc[-1].reshape(1, 1, -1)]*decoder.num_layers)
                    if config.cell_type == 'lstm':
                        cell_dec = torch.cat([cell_enc[-1].reshape(1, 1, -1)]*decoder.num_layers)

                    prediction = ''

                    max_length = MAX_LENGTH
                    while max_length > 0:
                        if config.cell_type == 'lstm':
                            output_dec, (hidden_dec, cell_dec), attn = decoder(input_dec, (hidden_dec, cell_dec), outputs_enc)
                        else:
                            output_dec, hidden_dec, attn = decoder(input_dec, hidden_dec, outputs_enc)
                        pred_char_index = output_dec.data.argmax()
                        prediction += tam_alphabet.index_to_letter[pred_char_index]
                        if pred_char_index.item() == 1:
                            break
                        input_dec = pred_char_index.squeeze().detach()
                        max_length -= 1
                    
                    if prediction == val_data_target[val_index]+'EOW':
                        val_score += 1.
                val_score /= len(val_data_given)
                print(f'Epoch {epoch+1}/{config.epochs} => Loss: {train_loss}, Train accuracy: {train_score}, Val score: {val_score}')
            
            wandb.log({'epoch': epoch+1, 'loss': train_loss, 'accuracy': train_score, 'val_accuracy': val_score})
            if prev_score >= val_score:
                break
            else:
                prev_score = val_score

sweep_config = {
    'method': 'bayes',
    'name': 'attention_sweep'
}

sweep_metric = {
    'name': 'val_accuracy',
    'goal': 'maximize'
}
sweep_config['metric'] = sweep_metric

parameters = {
    'cell_type': {
        'values': ['rnn', 'lstm', 'gru']
    },
    'embedding_size': {
        'values': [8, 16, 32]
    },
    'encoder_layers': {
        'values': [1, 2, 3]
    },
    'decoder_layers': {
        'values': [1, 2, 3]
    },
    'hidden_size': {
        'values': [64, 128, 256]
    },
    'dropout': {
        'values': [0.0, 0.1, 0.2, 0.3]
    },
    'lr': {
        'value': 1e-3
    },
    'weight_decay': {
        'values': [1e-5, 1e-3, 1e-1, 0.0]
    },
    'epochs': {
        'value': 10
    }
}
sweep_config['parameters'] = parameters

sweep_id = wandb.sweep(sweep_config, project='CS6910-A3')
# sweep_id = 'x953yurr'
wandb.agent(sweep_id, perform_run, project='CS6910-A3')
