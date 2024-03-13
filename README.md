# pytorch-transformer

## Introduction
A transformer model, as described in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762), is implemented in the PyTorch framework from scratch. The model is trained and validated for machine translation tasks for various datasets from [Hugging Face](https://huggingface.co/datasets).

## Getting Started
1. Clone the repository to the desired directory:
```bash
git clone https://github.com/brondaeh/pytorch-transformer
```
2. Install the required libraries to your environment:
```bash
pip install -r requirements.txt
```
3. Modify the following configuration parameters in config.py.
```python
def get_config():
    return {
        'datasource': 'dataset_name',           # dataset name from hugging face
        'lang_source': 'source_language_code',  # ISO 639-1 language code
        'lang_target': 'target_language_code',  # ISO 639-1 language code
        'preload': 'latest',                    # 'latest' to preload the most recent save; None to turn off preloading
    }
```
4. Start training and validation by running train.py.

## Trained Datasets
1. opus_books: en-it
2. opus_openoffice: en_GB-zh_CN

## References
1. [pytorch-transformer](https://github.com/hkproj/pytorch-transformer)
2. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
3. [Hugging Face Datasets](https://huggingface.co/datasets)