import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class CNNClassifier(nn.Module):
    def __init__(self, batch_size, output_size, out_channels, kernel_sizes, dropout, vocab_size, embedding_dim, weights, unfreeze_embeddings=False):
        """
        Arguments
        ---------
        batch_size : Size of each batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        in_channels : Number of input channels. Here it is 1 as the input data has dimension = (batch_size, num_seq, embedding_length)
        out_channels : Number of output channels after convolution operation performed on the input matrix (also number of filters)
        kernel_sizes : A list consisting of 3 different kernel_sizes. Convolution will be performed 3 times and finally results from each kernel_size will be concatenated.
        dropout : Probability of retaining an activation node during dropout operation
        vocab_size : Size of the vocabulary containing unique words
        embedding_dim : Embedding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table
        unfreeze_embeddings: Boolean flag to unfreeze the word embeddings weights
        --------
            
        """
        super(CNNClassifier, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        if weights is not None:
            self.word_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(weights))
            # self.word_embeddings.weight = nn.Parameter(weights) # using pre-trained weights
            self.word_embeddings.weight.requires_grad=unfreeze_embeddings
        else:
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels = 1,
                      out_channels = out_channels,
                      kernel_size = (embedding_dim, ks)
                     ) for ks in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        
        self.fc = nn.Linear(len(kernel_sizes) * out_channels, output_size)

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input) # conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3)) # activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2) # maxpool_out.size() = (batch_size, out_channels)
        
        return max_out

    def forward(self, input_sentences, batch_size=None):
        """
        The idea of the Convolutional Neural Netwok for Text Classification is very simple. We perform convolution operation on the embedding matrix 
        whose shape for each batch is (num_seq, embedding_dim) with kernel of varying height but constant width which is same as the embedding_length.
        We will be using ReLU activation after the convolution operation and then for each kernel height, we will use max_pool operation on each tensor 
        and will filter all the maximum activation for every channel and then we will concatenate the resulting tensors. This output is then fully connected
        to the output layers consisting two units which basically gives us the logits for both positive and negative classes.
        
        Parameters
        ----------
        input_sentences: input_sentences of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)
        
        Returns
        -------
        Output of the linear layer containing logits for pos & neg class.
        logits.size() = (batch_size, output_size)

        """
        #input_sentences = [sent len, batch size]
        text = input_sentences.permute(1, 0)
        #text = [batch size, sent len]
        embedded = self.embedding(text)
        #embedded = [batch size, sent len, emb dim]
        embedded = embedded.permute(0, 2, 1)
        #embedded = [batch size, emb dim, sent len]
        max_out = [self.conv_block(embedded, conv) for conv in self.convs]
        #max_out = [batch size, out_channels]
        all_out = torch.cat(max_out, dim=1)
        # all_out.size() = (batch_size, num_kernels*out_channels)
        fc_in = self.dropout(all_out)
        # fc_in.size()) = (batch_size, num_kernels*out_channels)
        logits = self.fc(fc_in)

        return logits