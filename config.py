from pathlib import Path

def get_config():
    return {
        'batch_size': 16,
        'num_epochs': 50,
        'lr': 10**-4,
        'seq_len': 350,
        'd_model': 512,
        'datasource': 'opus_openoffice',             # dataset name from hugging face
        'lang_source': 'en_GB',                    
        'lang_target': 'zh_CN',
        'model_folder': 'weights',
        'model_basename': 'tmodel_',
        'preload': 'latest',                    # None or 'latest'
        'tokenizer_file': 'tokenizer_{0}.json',
        'experiment_name': 'runs/tmodel'
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
