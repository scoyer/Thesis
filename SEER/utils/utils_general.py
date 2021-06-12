import torch
import torch.nn as nn
import torch.utils.data

from utils.config import *

from itertools import chain

def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    if max_len is None:
        max_len = lengths.max().item()
    mask = torch.arange(0, max_len, dtype=torch.long).type_as(lengths)
    mask = mask.unsqueeze(0)
    mask = mask.repeat(1, *lengths.size(), 1)
    mask = mask.squeeze(0)
    mask = mask.lt(lengths.unsqueeze(-1))
    #mask = mask.repeat(*lengths.size(), 1).lt(lengths.unsqueeze(-1))
    return mask


def to_onehot(labels, max_len=None, mask=None):
    """
    Create one hot vector given labels matrix.
    """
    if max_len is None:
        max_len = labels.max().item() + 1
    labels_size = labels.size()
    labels = labels.view(-1)
    onehot = torch.zeros(labels.size()+(max_len,)).type_as(labels).scatter_(1, labels.unsqueeze(1), 1).float()
    onehot = onehot.view(labels_size+(max_len,))
    if mask is not None:
        mask = mask.unsqueeze(-1).expand_as(onehot)
        onehot = onehot * (1.0 - mask.float())
    return onehot


def _cuda(x):
    if USE_CUDA:
        return x.cuda()
    else:
        return x

class Vocab:
    def __init__(self, is_for_attribute=False):
        if is_for_attribute:
            self.word2index = {}
            self.index2word = {}
            self.n_words = 0
        else:
            self.word2index = {}
            self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: 'UNK'}
            self.n_words = len(self.index2word) # Count default tokens
            self.word2index = dict([(v, k) for k, v in self.index2word.items()])
      
    def index_words(self, words):
        assert isinstance(words, (str,list))

        if isinstance(words, str):
            self.index_word(words)
        else:
            word_list = self.expand_list(words)
            for word in word_list:
                assert isinstance(word, str)
                self.index_word(word)

    def expand_list(self, multi_list):
        return_list = list(multi_list)
        while isinstance(return_list[0], list):
            return_list = sum(return_list, [])
        return return_list

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def delete_words(self, word_list):
        for word in word_list:
            if word in self.word2index:
                self.word2index.pop(word)

        remain_words = self.word2index.keys()
        self.__init__()
        for word in remain_words:
            self.index_word(word)


class Dataset(torch.utils.data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_info, word2id, word2id_attribute, map_table={}):
        """Reads source and target sequences from txt files."""
        self.data_info = {}
        for k in data_info.keys():
            self.data_info[k] = data_info[k]

        self.num_total_seqs = len(data_info['context'])
        self.word2id = word2id
        self.word2id_attribute = word2id_attribute
        self.map_table = map_table

    def preprocess(self, sequence, word2id, trg=True):
        """Converts words to ids."""
        #story = [word2id[word] if word in word2id else UNK_token for word in sequence]
        story = []
        for word in sequence:
            if word in self.map_table:
                word = self.map_table[word]
            if word in word2id:
                story.append(word2id[word])
            else:
                story.append(UNK_token)

        if trg:
            story = story + [EOS_token]

        story = torch.Tensor(story)
        return story
    
    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        context = self.preprocess(self.data_info['context'][index], self.word2id, trg=False)
        entity = self.preprocess(self.data_info['entity'][index], self.word2id, trg=False)
        response = self.preprocess(self.data_info['response'][index], self.word2id)
        sketch_response = self.preprocess(self.data_info['sketch_response'][index], self.word2id)

        context_entity_pos = torch.tensor(self.data_info['context_entity_pos'][index])
        context_entity_id = torch.tensor(self.data_info['context_entity_id'][index])
        response_entity_id = torch.tensor(self.data_info['response_entity_id'][index])
        indicator = torch.tensor(self.data_info['indicator'][index])

        kb_entity_id = sum(self.data_info['kb_entity_id'][index], [])

        # processed information
        data_info = {}
        for k in self.data_info.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = self.data_info[k][index]

        # additional plain information
        data_info['context_plain'] = " ".join(self.data_info['context'][index])
        data_info['entity_plain'] = self.data_info['entity'][index]
        data_info['response_plain'] = " ".join(self.data_info['response'][index])
        data_info['sketch_response_plain'] = " ".join(self.data_info['sketch_response'][index])
        return data_info

    def __len__(self):
        return self.num_total_seqs

    def collate_fn(self, data):
        def merge(sequences):
            lengths = torch.tensor([len(seq) for seq in sequences]).long()
            max_len = 1 if max(lengths)==0 else max(lengths)
            mask = torch.ones(len(sequences), max_len).byte()
            padded_seqs = torch.zeros(len(sequences), max_len).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
                mask[i,:end] = torch.zeros(end)
            return padded_seqs, lengths, mask

        def merge_kb(sequences):
            # sequences is a list that contains entity list
            # entity list is a list of tuple with format (index in entity set, row index in table, column attribute)

            batch_size = len(sequences)
            kb_entity_lengths = torch.tensor([len(seq) for seq in sequences]).long()
            max_len = max(kb_entity_lengths)
            kb_entity_id = torch.zeros(batch_size, max_len).long()
            kb_entity_mask = torch.ones(batch_size, max_len).long()
            kb_entity_row = torch.zeros(batch_size, max_len).long()
            kb_entity_col = torch.zeros(batch_size, max_len).long()
            kb_entity_nei = torch.zeros(batch_size, max_len, max_len).long()
            for b in range(batch_size):
                kb = sequences[b]
                kb_dict = {}
                for kb_id, (index, row_index, col_attribute) in enumerate(kb):
                     kb_entity_id[b, kb_id] = index
                     kb_entity_row[b, kb_id] = row_index
                     kb_entity_col[b, kb_id] = self.word2id_attribute[col_attribute]
                     if row_index not in kb_dict: kb_dict[row_index] = []
                     kb_dict[row_index].append(kb_id)
                for kb_id, (index, row_index, col_attribute) in enumerate(kb):
                    neighbor = torch.zeros(max_len)
                    for nei_id in kb_dict[row_index]:
                        if nei_id != kb_id:
                            neighbor[nei_id] = 1
                    kb_entity_nei[b, kb_id] = neighbor
                kb_entity_mask[b, :len(kb)] = torch.zeros(len(kb))
            return kb_entity_id, kb_entity_row, kb_entity_col, kb_entity_nei, kb_entity_lengths, kb_entity_mask

        item_info = {}
        for key in data[0].keys():
            item_info[key] = [d[key] for d in data]

        # merge sequences
        context, context_lengths, context_mask = merge(item_info['context'])
        entity, entity_lengths, entity_mask = merge(item_info['entity'])
        response, response_lengths, response_mask = merge(item_info['response'])
        sketch_response, _, _ = merge(item_info['sketch_response'])

        context_entity_pos, context_entity_lengths, context_entity_mask = merge(item_info['context_entity_pos'])
        context_entity_id, _, _ = merge(item_info['context_entity_id'])
        response_entity_id, _, _ = merge(item_info['response_entity_id'])
        indicator, _, _ = merge(item_info['indicator'])
        indicator = indicator.float()

        # generate
        # kb_entity_id  : B x maxE 
        # kb_entity_col : B x maxE
        # kb_entity_row : B x maxE
        # kb_entity_nei : B x maxE x maxE
        kb_entity_id, kb_entity_row, kb_entity_col, kb_entity_nei, kb_entity_lengths, kb_entity_mask = merge_kb(item_info['kb_entity_id'])

        # processed information
        data_info = {}
        for k in item_info.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = item_info[k]

        data_info['kb_entity_row'] = kb_entity_row
        data_info['kb_entity_col'] = kb_entity_col
        data_info['kb_entity_nei'] = kb_entity_nei

        data_info['context_mask'] = context_mask
        data_info['entity_mask'] = entity_mask
        data_info['response_mask'] = response_mask
        data_info['kb_entity_mask'] = kb_entity_mask
        data_info['context_entity_mask'] = context_entity_mask

        data_info['context_lengths'] = context_lengths
        data_info['entity_lengths'] = entity_lengths
        data_info['response_lengths'] = response_lengths
        data_info['kb_entity_lengths'] = kb_entity_lengths
        data_info['context_entity_lengths'] = context_entity_lengths

        # convert to contiguous and cuda
        fliter_list = ['ent_', 'dialog_', 'domain', '_plain', '_type']
        for k in data_info.keys():
            fliter = False
            for fliter_word in fliter_list:
                if fliter_word in k:
                    fliter = True
            if not fliter:
                data_info[k] = _cuda(data_info[k].contiguous())
        return data_info

def get_seq(pairs, vocab, vocab_attribute, batch_size, training=False, map_table={}):
    data_info = {}
    for k in pairs[0].keys():
        data_info[k] = []
    
    for pair in pairs:
        for k in pair.keys():
            data_info[k].append(pair[k])
    
    dataset = Dataset(data_info, vocab.word2index, vocab_attribute.word2index, map_table=map_table)
    data_loader = torch.utils.data.DataLoader(dataset = dataset,
                                              batch_size = batch_size,
                                              shuffle = training,
                                              collate_fn = dataset.collate_fn)
    return data_loader
