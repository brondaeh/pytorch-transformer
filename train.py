import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_weights_file_path, get_config, latest_weights_file_path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter

import warnings
from tqdm import tqdm
from pathlib import Path


def greedy_decode(model, source, source_mask, tokenizer_source, tokenizer_target, max_len, device):
    '''
    Inference strategy which chooses the next token to be the one with the highest probability from the projection layer

    Args:
    - model (Transformer): instance of Transformer class
    - source (torch.tensor): encoder input sequence
    - source_mask (torch.tensor): source sequence mask
    - tokenizer_source (Tokenizer): tokenizer to process source sequence
    - tokenizer_target (Tokenizer): tokenizer to process target sequence
    - max_len (int): maximum length of the generated sequence
    - device (torch.device): device used for tensor computations

    Return:
    - decoder_input (torch.tensor): the generated sequence
    '''
    sos_idx = tokenizer_source.token_to_id('[SOS]')
    eos_idx = tokenizer_target.token_to_id('[EOS]')

    # Precalculate encoder output to be resused for each together recieved from the decoder
    encoder_output = model.encode(source, source_mask)
    
    # Initialize decoder input with SOS token and generate the next token until max_len or EOS token is reached
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # Create mask for the target (decoder input)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get the next token
        prob = model.project(out[:, -1])

        # Choose the token with the largest probability (greedy search)
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break
    
    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_source, tokenizer_target, max_len, device, print_msg, global_step, writer, num_examples=2):
    '''
    Evaluates the trained model

    Args:
    - model (Transformer): instance of Transformer class
    - validation_ds: validation dataset
    - tokenizer_source (Tokenizer): tokenizer to process the source sequence
    - tokenizer_target (Tokenizer): tokenizer to process the target sequence
    - max_len (int): maximum length of the generated sequence
    - device (torch.device): device used for tensor computations
    - print_msg: function used to print messages
    - global_step and writer: excluded, used for logging data during training and validation
    - num_examples (int): number of examples to show during validation

    Return: None
    '''
    model.eval()    # set model to evaluation mode
    count = 0       # track the number of batches of examples processed during validation

    # source_texts = []
    # expected = []
    # predicted = []

    # Size of the control window
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)   # (batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)     # (batch, 1, 1, seq_len)

            # Check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_source, tokenizer_target, max_len, device)

            source_text = batch['source_text'][0]
            target_text = batch['target_text'][0]
            model_out_text = tokenizer_target.decode(model_out.detach().cpu().numpy())

            # source_texts.append(source_text)
            # expected.append(target_text)
            # predicted.append(model_out_text)

            # Print to console
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break

    # if writer:
    #     # Evaluate the character error rate
    #     # Compute the char error rate 
    #     metric = torchmetrics.CharErrorRate()
    #     cer = metric(predicted, expected)
    #     writer.add_scalar('validation cer', cer, global_step)
    #     writer.flush()

    #     # Compute the word error rate
    #     metric = torchmetrics.WordErrorRate()
    #     wer = metric(predicted, expected)
    #     writer.add_scalar('validation wer', wer, global_step)
    #     writer.flush()

    #     # Compute the BLEU metric
    #     metric = torchmetrics.BLEUScore()
    #     bleu = metric(predicted, expected)
    #     writer.add_scalar('validation BLEU', bleu, global_step)
    #     writer.flush()

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    # config['tokenizer_file'] = '../tokenizers/tokenizer_{0}.json'
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    '''
    Args:
    - config (dictionary): specified model parameters

    Return:
    - train_dataloader, val_dataloader, tokenizer_source, tokenizer_target (tuple): both dataloaders and tokenizers of source and target languages
    '''
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_source']}-{config['lang_target']}", split='train')

    # Build tokenizers
    tokenizer_source = get_or_build_tokenizer(config, ds_raw, config['lang_source'])
    tokenizer_target = get_or_build_tokenizer(config, ds_raw, config['lang_target'])

    # Keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_source, tokenizer_target, config['lang_source'], config['lang_target'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_source, tokenizer_target, config['lang_source'], config['lang_target'], config['seq_len'])

    # Find the max length of the source and target sentences
    max_len_source = 0
    max_len_target = 0

    for item in ds_raw:
        source_ids = tokenizer_source.encode(item['translation'][config['lang_source']]).ids
        target_ids = tokenizer_target.encode(item['translation'][config['lang_target']]).ids
        max_len_source = max(max_len_source, len(source_ids))
        max_len_target = max(max_len_target, len(target_ids))

    print(f'Max length of source sentence: {max_len_source}')
    print(f'Max length of target sentence: {max_len_target}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_source, tokenizer_target

def get_model(config, vocab_source_len, vocab_target_len):
    '''
    Args:
    - config (dictionary): specified model parameters
    - vocab_source_len (int): vocabulary size for the source language
    - vocab_target_len (int): vocabulary size for the target language

    Return:
    - model (Transformer): a Transformer instance
    '''
    model = build_transformer(vocab_source_len, vocab_target_len, config['seq_len'], config['seq_len'], d_model=config['d_model'])
    return model

def train_model(config):
    '''
    Trains the model and validates the model at the end of each training epoch

    Args:
    - config (dictionary): specified model parameters

    Return: None
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Chosen device: {device}')

    # Ensure the corresponding weights folder exists for the datasource
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    # Initialize datasets, tokenizers, and transformer model
    train_dataloader, val_dataloader, tokenizer_source, tokenizer_target = get_ds(config)
    model = get_model(config, tokenizer_source.get_vocab_size(), tokenizer_target.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer_source.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config['num_epochs'])

    # If specified, preload the model before training
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing Epoch {epoch:02d}')
        
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)   # (batch, seq_len)
            decoder_input = batch['decoder_input'].to(device)   # (batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)     # (batch, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)     # (batch, 1, seq_len, seq_len)

            # Run tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask)                                  # (batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)    # (batch, seq_len, d_model)
            proj_output = model.project(decoder_output)                                                 # (batch, seq_len, target_vocab_size)

            # Compare output with label
            label = batch['label'].to(device)   # (batch, seq_len)

            # Compute the loss: (batch, seq_len, target_vocab_size) -> (batch * seq_len, target_vocab_size)
            loss = criterion(proj_output.view(-1, tokenizer_target.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({'loss': f'{loss.item():6.3f}'})

            # Apply log to the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropogate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # Run validation at the end of each epoch
        run_validation(model, val_dataloader, tokenizer_source, tokenizer_target, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
        scheduler.step()
        
        # Save the model for every 10 epochs
        if epoch % 10 == 0:
            model_filename = get_weights_file_path(config, f'{epoch:02d}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()           # retrieve the training parameters
    train_model(config)             # train the model 
