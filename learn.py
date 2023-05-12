from seq2seq import Encoder, Decoder, AttnDecoder
from data import word_to_tensor, Alphabet
from typing import Union, List, Tuple
import torch

def train(
    encoder: Encoder,
    decoder: Union[Decoder, AttnDecoder],
    X: List[str],
    Y: List[str],
    alphabets: Tuple[Alphabet, Alphabet],
    optimizer_enc: torch.optim.Optimizer,
    optimizer_dec: torch.optim.Optimizer,
    loss_fn: torch.nn.Module = torch.nn.NLLLoss(),
    max_length: int = 100,
    teacher_forcing_ratio: float = 0.5,
    device: torch.device = 'cpu'
) -> Tuple[float, float]:
    total_loss, total_score = 0.0, 0.0
    for train_index in range(len(X)):
        hidden_enc = encoder.initHidden().to(device, non_blocking=True)
        cell_enc = encoder.initHidden().to(device, non_blocking=True)

        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()

        input_vector = word_to_tensor(alphabets[0], X[train_index]).to(device, non_blocking=True)
        input_len = len(input_vector)

        outputs_enc = torch.zeros(max_length, encoder.hidden_size).to(device, non_blocking=True)

        loss = 0.0

        for i, char in enumerate(input_vector):
            if encoder.is_lstm:
                output_enc, (hidden_enc, cell_enc) = encoder(char, (hidden_enc, cell_enc))
            else:
                output_enc, hidden_enc = encoder(char, hidden_enc)
            outputs_enc[i] = output_enc[0, 0]
            
        input_dec = torch.tensor([[0]]).to(device, non_blocking=True)
        target_vector = word_to_tensor(alphabets[1], Y[train_index]).to(device, non_blocking=True)
        target_len = len(target_vector)

        hidden_dec = torch.cat([hidden_enc[-1].reshape(1, 1, -1)]*decoder.num_layers).to(device, non_blocking=True)
        if encoder.is_lstm:
            cell_dec = torch.cat([cell_enc[-1].reshape(1, 1, -1)]*decoder.num_layers).to(device, non_blocking=True)

        tfr = torch.rand(1) < teacher_forcing_ratio
        prediction = ''

        for di in range(target_len):
            if decoder.is_lstm:
                if type(decoder) == AttnDecoder:
                    output_dec, (hidden_dec, cell_dec), attn = decoder(input_dec, (hidden_dec, cell_dec), outputs_enc)
                else:
                    output_dec, (hidden_dec, cell_dec) = decoder(input_dec, (hidden_dec, cell_dec))
            else:
                if type(decoder) == AttnDecoder:
                    output_dec, hidden_dec, attn = decoder(input_dec, hidden_dec, outputs_enc)
                else:
                    output_dec, hidden_dec = decoder(input_dec, hidden_dec)
            if tfr:
                loss += loss_fn(output_dec, target_vector[di])
                input_dec = target_vector[di]
                prediction += alphabets[1].index_to_letter[output_dec.argmax(dim=1).squeeze().detach()]
            else:
                loss += loss_fn(output_dec, target_vector[di])
                input_dec = output_dec.argmax(dim=1).squeeze().detach()
                prediction += alphabets[1].index_to_letter[input_dec]
                if input_dec.item() == 1:
                    break
        
        loss.backward()
        optimizer_enc.step()
        optimizer_dec.step()

        total_loss += loss.detach() / input_len
        if prediction == Y[train_index]+'EOW':
            total_score += 1.
    
    total_loss /= len(X)
    total_score /= len(Y)

    return total_loss, total_score
                
def predict(
    encoder: Encoder,
    decoder: Union[Decoder, AttnDecoder],
    X: List[str],
    alphabets: Tuple[Alphabet, Alphabet],
    max_length: int = 100,
    device: torch.device = 'cpu'
) -> Tuple[List[str], List]:
    attentions, predictions = [], []
    with torch.inference_mode():
        for val_index in range(len(X)):
            word_attns = []
            input_vector = word_to_tensor(alphabets[0], X[val_index]).to(device, non_blocking=True)
            input_len = len(input_vector)
            hidden_enc = encoder.initHidden().to(device, non_blocking=True)
            cell_enc = encoder.initHidden().to(device, non_blocking=True)
            outputs_enc = torch.zeros(max_length, encoder.hidden_size).to(device, non_blocking=True)

            for i, char in enumerate(input_vector):
                if encoder.is_lstm:
                    output_enc, (hidden_enc, cell_enc) = encoder(char, (hidden_enc, cell_enc))
                else:
                    output_enc, hidden_enc = encoder(char, hidden_enc)
                outputs_enc[i] = output_enc[0, 0]
            
            input_dec = torch.tensor([[0]]).to(device, non_blocking=True)
            hidden_dec = torch.cat([hidden_enc[-1].reshape(1, 1, -1)]*decoder.num_layers)
            if encoder.is_lstm:
                cell_dec = torch.cat([cell_enc[-1].reshape(1, 1, -1)]*decoder.num_layers)

            prediction = ''
            rem_length = max_length

            while rem_length > 0:
                if decoder.is_lstm:
                    if type(decoder) == AttnDecoder:
                        output_dec, (hidden_dec, cell_dec), attn = decoder(input_dec, (hidden_dec, cell_dec), outputs_enc)
                        word_attns.append(attn)
                    else:
                        output_dec, (hidden_dec, cell_dec) = decoder(input_dec, (hidden_dec, cell_dec))
                else:
                    if type(decoder) == AttnDecoder:
                        output_dec, hidden_dec, attn = decoder(input_dec, hidden_dec, outputs_enc)
                        word_attns.append(attn)
                    else:
                        output_dec, hidden_dec = decoder(input_dec, hidden_dec)
                pred_char_index = output_dec.data.argmax()
                prediction += alphabets[1].index_to_letter[pred_char_index]
                if pred_char_index.item() == 1:
                    break
                input_dec = pred_char_index.squeeze().detach()
                rem_length -= 1
            
            if type(decoder) == AttnDecoder:
                attentions.append(word_attns)
            predictions.append(prediction)
    
    return predictions, attentions
