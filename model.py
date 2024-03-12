import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # Map the indices of the input sequence of tokens (x) to corresponding word embeddings
        # Scale by sqrt(d_model) so that embeddings have reasonably sized values
        # Input x (batch, seq_len) -> Output embedding (batch, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model) to store positional encodings
        pe = torch.zeros(seq_len, d_model)
        
        # Create a vector of shape (seq_len, 1) containing values from 0 to seq_len - 1
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # Create a vector of shape (d_model / 2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)    # sin(position * (10000 ** (2i / d_model))

        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)    # cos(position * (10000 ** (2i / d_model))

        # Add a batch dimension to positional encoding matrix
        pe = pe.unsqueeze(0)    # (1, seq_len, d_model)

        # PEs are not learned parameters of the model, but these tensors must be stored in a buffer to be reused for different inputs
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input x
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)  # requires_grad_(False) is used so the tensor is not learned by the model
        return self.dropout(x)
    

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps                                      # epsilon is used for numerical stability and to avoid division by 0
        self.alpha = nn.Parameter(torch.ones(features))     # learned parameter that is multiplied
        self.bias = nn.Parameter(torch.zeros(features))     # learned parameter that is added

    def forward(self, x):
        # Shape x: (batch, seq_len, hidden_size)
        mean = x.mean(dim = -1, keepdim=True)       # (batch, seq_len, 1)
        std = x.std(dim = -1, keepdim=True)         # (batch, seq_len, 1)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)    # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)    # W2 and B2

    def forward(self, x):
        # Input (batch, seq_len, d_model) -> Intermediate (batch, seq_len, d_ff) -> Output (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model (embedding dimensions) is not divisible by h (# of heads)"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)  # WQ
        self.w_k = nn.Linear(d_model, d_model)  # WK
        self.w_v = nn.Linear(d_model, d_model)  # WV
        self.w_o = nn.Linear(d_model, d_model)  # WO where, h * d_v = h * d_k = d_model
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        '''
        Computes the single-head attention matrix for each submatrix split by the number of heads

        Args:
        - query, key, value (torch.tensor): submatrices (batch, h, seq_len, d_k
        - dropout (nn.Dropout): dropout layer

        Return:
        - (attention_scores @ value, attention_scores): tuple of attention matrix and attention scores
        '''
        d_k = query.shape[-1]

        # Input (batch, h, seq_len, d_k) -> Output (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)      # replace (mask) values that are 0 for -1e9 (negative inf)
        attention_scores = attention_scores.softmax(dim=-1)   # (batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores     # return tuple of attention matrix and model scores for visualization

    def forward(self, q, k, v, mask):
        # Multiply input matrices Q, K, V (batch, seq_len, d_model) by W_Q, W_K, W_V (d_model, d_model) -> Q', K', V' (batch, seq_len, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Split matrices based on number of heads: (batch, seq_len, d_model) -> (batch, seq_len, d_k) -> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        #(batch, seq_len, d_model) -> (batch, seq_len, d_model)
        return self.w_o(x)
    

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # Lambda is used to apply self-attention to the input x where the queries, keys, and values are all x; the result is passed to the skip connection
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        # The feed-forward block is applied to the input x and the result is passed to the second skip connection
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    

class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
    

class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range (3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.self_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    

class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # Input (batch, seq_len, d_model) -> Output (batch, seq_len, vocab_size)
        return self.project(x)
    

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embed = src_embed
        self.target_embed = tgt_embed
        self.source_pos = src_pos
        self.target_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.source_embed(src)
        src = self.source_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.tensor, src_mask: torch.tensor, tgt: torch.tensor, tgt_mask: torch.tensor):
        tgt = self.target_embed(tgt)
        tgt = self.target_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    '''
    Builds the transformer model

    Args:
    - src_vocab_size (int): size of the source language vocabulary of unique tokens
    - tgt_vocab_size (int): size of the target language vocabulary of unique tokens
    - src_seq_len (int): maximum sequence length of the source language input
    - tgt_seq_len (int): maximum sequence length of the target language input
    - d_model (int = 512): model dimensionality; the embedding vector size
    - N (int = 6): the number of encoder and decoder layers; the number of times self-attention and feed-forward is applied
    - h (int = 8): the number of heads used in multi-head attention
    - dropout (float = 0.1): the dropout probability used for regularization; prevents overfitting by ignoring random neurons
    - d_ff (int = 2048): feed-forward layer dimensionality; the size of hidden layer in the feed-forward block

    Return: 
    - transformer (Transformer): a Transformer instance
    '''
    # Create embedding layers: converts token indices to embeddings for both source and target languages
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create positional emcoding layers (source_pos = target_pos): provides positional info to the embeddings
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create encoder blocks: a list of N encoder blocks each with a self-attention and feed-forward layer
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create decoder blocks: a list of N decoder blocks each with a self-attention, cross-attention, and feed-forward layer
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create encoder and decoder: instances of Encoder and Decoder classes with the previously defined blocks
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create projection layer: projects decoder output into the vocabulary space
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create transformer: an instance of the Transformer class
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize parameters
    for p in transformer.parameters():  # iterate over all transformer model parameters, each parameter is a tensor
        if p.dim() > 1:                 # check if the parameter has more than 1 dimension -> weight tensor (not bias tensor)
            nn.init.xavier_uniform_(p)  # initialize the tensor p with values from the xavier uniform distribution

    return transformer
