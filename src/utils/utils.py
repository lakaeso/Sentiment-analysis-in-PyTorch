from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

import numpy as np

from typing import Any


@dataclass
class Instance:
    """
    Represents a shallow wrapper of tokens and labels
    """
    tokens: list[int]
    label: int 
    
    def __iter__(self):
        return iter([self.tokens, self.label])
    
    
class NLPDataset(Dataset):
    """Abstracts data which will be used by our models. Instances of NLPDataset will be passed to DataLoader.
    
    Implements __getitem__ and __len__."""
    
    def __init__(self, path_to_data: str, max_size = -1, min_freq = -1, path_to_embedding: str = './data/sst_glove_6b_300d.txt', text_vocab: 'Vocab' = None) -> None:
        super().__init__()
        
        frequencies = {}
        self.instances = []
        
        with open(path_to_data, 'r') as f:
            content = f.read()
            
            for line in content.split('\n'):
                if line == '':
                    continue
                
                tokens: list[str] = line.split(', ')[0].split(' ')
                label: str = line.split(', ')[1]
                
                self.instances.append((tokens, label))
                
                for word in tokens:
                    if word in frequencies:
                        frequencies[word] += 1
                    else:
                        frequencies[word] = 1
                        
        # set text vocab if needed
        if text_vocab is None:
            self.text_vocab = Vocab(frequencies, max_size=max_size, min_freq=min_freq)
        else:
            self.text_vocab = text_vocab
            
        self.label_vocab = LabelVocab()
        self.max_size = max_size
        self.min_freq = min_freq
        
        # set embedding
        embeddings_raw = {}
        with open(path_to_embedding, 'r') as f:
            lines = f.read().split("\n")
            
            for line in lines:
                if line == '':
                    continue
                label = line.split(' ', 1)[0]
                values = line.split(' ', 1)[1]
                embeddings_raw[label] = values
        
        # create embedding matrix
        N = len(self.text_vocab.stoi)
        d = 300

        embedding = np.random.normal(loc=0, scale=1, size=(N, d))
        
        for token, position in self.text_vocab.stoi.items():
            if token == "<PAD>":
                embedding[0] = np.zeros(d)
            if token == "<UNK>":
                # embedding[1] = np.zeros(d)
                pass
            if token in embeddings_raw:
                nums = embeddings_raw[token].split(' ')
                nums = [float(i) for i in nums]
                embedding[position] = torch.tensor(nums, dtype=torch.float32)
        
        embedding = torch.tensor(embedding.astype(np.float32))
        
        self.embedding = torch.nn.Embedding.from_pretrained(embedding, freeze=True, padding_idx=0)
            
    def __getitem__(self, index) -> Instance:
        # fetch
        tokens = self.instances[index][0]
        label = self.instances[index][1]
        
        # transform
        tokens = self.text_vocab.encode(tokens)
        label = self.label_vocab.encode(label)
        
        return Instance(tokens, label)
    
    def __len__(self):
        return len(self.instances)
    
class BERTDataset(Dataset):
    def __init__(self, path_to_data, tokenizer):    
        super()
        
        self.tokenizer = tokenizer
        
        content = []
        with open(path_to_data, 'r') as f:
            file_rows = f.read().strip().lower().split('\n')
            
            for row in file_rows:
                [text, label] = row.split(',')
                label = label.strip()
                
                tokens = tokenizer.tokenize(text) 
                tokens = tokens[:512-2]
                
                content.append((tokens, label))
        
        self.instances = content
        
    def __getitem__(self, index) -> Instance:
        # fetch
        tokens = self.instances[index][0]
        label = self.instances[index][1]
        
        # transform
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        label = 1 if label == 'positive' else 0 # adhoc encoder
        
        return Instance(tokens, label)
    
    def __len__(self):
        return len(self.instances)
    
class Vocab:
    """Represents an NLP vocabulary. Can encode ('string-to-int' operation) a list of strings."""
    
    def __init__(self, frequencies: dict, max_size = -1, min_freq = -1) -> None:
        
        # tokens sorted by freqs
        sorted_freqs = sorted(frequencies, key=frequencies.get, reverse=True)
        
        # check and apply min_freq filter
        if min_freq > -1:
            sorted_freqs = list(filter(lambda x: frequencies[x] > min_freq, sorted_freqs))
        
        # check and apply max_size filter
        if max_size > -1:
            if max_size < len(sorted_freqs):
                sorted_freqs = sorted_freqs[:max_size]
        
        # build stoi dict
        self.stoi: dict[str, int] = {}
        
        # PAD and UNK tokens always present
        self.stoi["<PAD>"] = 0
        self.stoi["<UNK>"] = 1
        
        # for loop
        i = 2
        for token in sorted_freqs:
            self.stoi[token] = i
            i += 1

        # build itos
        self.itos: dict[int, str] = {}
        for key in self.stoi:
            self.itos[self.stoi[key]] = key
    
    def encode(self, tokens: list[str]) -> list[int]:
        """Encodes a list of strings (tokens) into a list of ints."""
        
        output = []
        tmp = 0
        
        for token in tokens:
            if token in self.stoi:
                tmp = self.stoi[token]
            else:
                tmp = self.stoi["<UNK>"]
            output.append(tmp)
            
        return output
    
    def decode(self):
        pass



class LabelVocab:
    """Represents a vocabulary for labels."""
    
    def __init__(self) -> None:
        self.stoi = {'positive': 0, 'negative': 1}
        self.itos = {0: 'positive', 1: 'negative'}
    
    def encode(self, label: str):
        assert label in self.stoi
        
        return self.stoi[label]
    
    def decode(self, num):
        assert num in self.itos
        
        return self.itos in self.itos
    
    
    
def pad_collate_fn(batch, pad_index=0):
    """Pads and converts input data to tensors and also adds info about their respective lengths.
    
    Returns a triplet in a form of (texts, labels, lengths)."""
    
    texts, labels = zip(*batch)
    lengths = [len(text) for text in texts]
    
    # pad text
    texts = list([torch.tensor(text) for text in texts])    
    texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=pad_index)
    
    # convert labels
    labels = torch.tensor(np.array(labels)).float()
    
    # convert lengths
    lengths = torch.tensor(lengths)
    
    # return
    return texts, labels, lengths