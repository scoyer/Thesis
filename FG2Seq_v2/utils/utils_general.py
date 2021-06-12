import math
import torch
import torch.utils.data as data
import torch.nn as nn
from utils.config import *

from itertools import chain
from collections import defaultdict

def get_schdule_sampling_rate(epoch, ratio):
    ss_rate = ratio / (ratio + math.exp(epoch / ratio) - 1)
    ss_rate = max(ss_rate, 0.5)
    return ss_rate


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

def mask_and_length(sequence, pad_idx):
    sequence_mask = (sequence != pad_idx)
    sequence_length = sequence_mask.long().sum(-1)
    sequence_mask = (1 - sequence_mask.long()).to(sequence_mask.dtype)
    return sequence_mask, sequence_length


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

def _cuda(x):
    if USE_CUDA:
        return x.cuda()
    else:
        return x

class Lang:
    def __init__(self, init=True):
        self.word2index = {}
        self.index2word = \
                {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: 'UNK'} if init else {PAD_token: "PAD"}
        self.n_words = len(self.index2word) # Count default tokens
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])
      
    def index_words(self, story, trg=False):
        if trg:
            for word in story:
                self.index_word(word)
        else:
            for word_triple in story:
                for word in word_triple:
                    self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_info, lang):
        self.vocabulary = lang[0]
        self.relation = lang[1]

        """Reads source and target sequences from txt files."""
        self.data_info = {}
        for k in data_info.keys():
            self.data_info[k] = data_info[k]

        self.num_total_seqs = len(data_info['context_arr'])
        self.src_word2id = self.vocabulary.word2index
        self.trg_word2id = self.vocabulary.word2index
        self.relation_size = self.relation.n_words
    
    def __getitem__(self, index):
        # processed information
        data_info = {}
        for k in self.data_info.keys():
            data_info[k] = self.data_info[k][index]

        return data_info

    def __len__(self):
        return self.num_total_seqs

    def collate_fn(self, data):
        def merge_sequence(sequences, fill_value=PAD_token, dtype=torch.long):
            max_len = max([len(seq) for seq in sequences])
            max_len = max_len if max_len > 0 else 1

            padded_seqs = torch.zeros(len(sequences), max_len).fill_(fill_value)
            for i, seq in enumerate(sequences):
                padded_seqs[i, :len(seq)] = torch.Tensor(seq)

            padded_seqs = padded_seqs.to(dtype)
            return padded_seqs

        def merge_array(sequences, fill_value=PAD_token, dtype=torch.long):
            dim1 = max([len(seq) for seq in sequences])
            dim1 = dim1 if dim1 > 0 else 1
            dim2 = max([len(s) for seq in sequences for s in seq])
            dim2 = dim2 if dim2 > 0 else 1

            padded_seqs = torch.zeros(len(sequences), dim1, dim2).fill_(fill_value)
            
            for i, seq in enumerate(sequences):
                for j, s in enumerate(seq):
                    padded_seqs[i, j, :len(s)] = torch.Tensor(s)

            padded_seqs = padded_seqs.to(dtype)
            
            return padded_seqs

        def merge_graph(sequences, edge_num, node_num):
            sequences = [torch.Tensor(seq) for seq in sequences]
            all_indices = torch.cat(sequences, dim=0)
            i = torch.zeros(all_indices.size(0), 4).long()
            i[:, 1:] = all_indices
            
            idx = 0
            for seq_id, seq in enumerate(sequences):
                i[idx:idx+seq.size(0), 0] = torch.LongTensor([seq_id] * seq.size(0))
                idx = idx + seq.size(0)

            v = torch.ones(i.size(0)).float()
            padded_seqs = torch.sparse.FloatTensor(i.t(), v, torch.Size([len(sequences), edge_num, node_num, node_num]))

            return padded_seqs

        item_info = {}
        for key in data[0].keys():
            item_info[key] = [d[key] for d in data]

        # merge sequences 
        context_arr = merge_array(item_info["context_arr"])


        kb_ent = merge_sequence(item_info["kb_ent"])
        kb_rel = merge_array(item_info["kb_rel"])
        kb_obj = merge_array(item_info["kb_obj"])
        kb_f = merge_array(item_info["kb_f"], fill_value=0,dtype=torch.float)
        kb_m = merge_array(item_info["kb_m"], fill_value=0,dtype=torch.uint8)

        sketch_response = merge_sequence(item_info['sketch_response'])
        ptr_index = merge_sequence(item_info['ptr_index'])
        graph = merge_graph(item_info['graph'], self.relation_size, kb_ent.size(1))

        # processed information
        data_info = {}
        for k in item_info.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = item_info[k]

            if isinstance(data_info[k], torch.Tensor):
                data_info[k] = _cuda(data_info[k])

        return data_info


def generate_indicator(context_arr, entity_list):
    """
    generate a list with the same size of context_arr, indicating whether each element of context_arr appears in kb_arr
    """

    indicator = []
    for question in context_arr:
        indicator.append([1 if entity in question else 0 for entity in entity_list])

    return indicator


def generate_index_from_kb(kb_arr, entity_list, relation):
    kb_ent = entity_list
    kb_rel = [[] for _ in range(len(kb_ent))]
    kb_obj = [[] for _ in range(len(kb_ent))]


    for kb in kb_arr:
        sub, rel, obj = kb[:3]
    
        sub_id = kb_ent.index(sub)
        rel_id = relation.word2index[rel]
        obj_id = kb_ent.index(obj)

        kb_rel[sub_id].append(rel_id)
        kb_obj[sub_id].append(obj_id)

        kb_rel[obj_id].append(rel_id)
        kb_obj[obj_id].append(sub_id)
    """
    # index 0 represent [cls]
    for ent_id in range(1, len(entity_list)):
        kb_rel[0].append(1)
        kb_obj[0].append(ent_id)

        kb_rel[ent_id].append(1)
        kb_obj[ent_id].append(0)
    """
    return kb_ent, kb_rel, kb_obj


def generate_graph(kb_arr, entity_list, relation):
    graph = []
    for kb in kb_arr:
        sub, rel, obj = kb[:3]
        sub_id = entity_list.index(sub)
        rel_id= relation.word2index[rel]
        obj_id = entity_list.index(obj)
        graph.append([rel_id, sub_id, obj_id])
        graph.append([rel_id, obj_id, sub_id])
    """
    if len(graph) == 0:
        graph.append([0, 0, 0])
    """
    # index 0 represent [cls]
    for ent_id in range(1, len(entity_list)):
        graph.append([0, 0, ent_id])
        graph.append([0, ent_id, 0])

    if len(graph) == 0:
        graph.append([0, 0, 0])
    
    return graph


def get_seq(pairs, lang, batch_size, is_shuffle):

    def to_vec(seq, vocab, triple=False, unk="UNK"):
        vec = []
        if not triple:
            for word in seq:
                vec.append(vocab[word] if word in vocab else vocab[unk])
        else:
            for sent in seq:
                vec.append([])
                for word in sent:
                    vec[-1].append(vocab[word] if word in vocab else vocab[unk])

        return vec
    
    (vocabulary, relation) = lang

    data_info = defaultdict(list)
    for pair in pairs:
        # entity related
        context_arr = pair["ct_arr"]
        entity_list = pair["entity_list"]
        kb_arr = pair["kb_arr"]
        kb_ent, kb_rel, kb_obj = generate_index_from_kb(kb_arr, entity_list, relation)
        kb_f = generate_indicator(context_arr, kb_ent)
        kb_m = [[1] * len(kb_rel[k]) for k in range(len(kb_rel))]
        graph = generate_graph(kb_arr, kb_ent, relation)

        # Vectoriztion
        context_arr = to_vec(context_arr, vocabulary.word2index, True)
        kb_ent = to_vec(kb_ent, vocabulary.word2index, False)
        sketch_response = to_vec(pair["sketch_response"], vocabulary.word2index, False)

        data_info["context_arr"].append(context_arr)
        data_info["kb_ent"].append(kb_ent)
        data_info["kb_rel"].append(kb_rel)
        data_info["kb_obj"].append(kb_obj)
        data_info['kb_f'].append(kb_f)
        data_info['kb_m'].append(kb_m)
        data_info["sketch_response"].append(sketch_response)
        data_info["graph"].append(graph)

        has_key = ["context_arr", "kb_ent", "kb_rel", "kb_obj", \
                        "sketch_response", "indicator", "graph"]
        for k in pair.keys():
            if k not in has_key:
                data_info[k].append(pair[k])

    
    dataset = Dataset(data_info, lang)
    data_loader = torch.utils.data.DataLoader(dataset = dataset,
                                              batch_size = batch_size,
                                              shuffle = is_shuffle,
                                              collate_fn = dataset.collate_fn)
    return data_loader
