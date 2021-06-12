import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from utils.config import *
from torch.nn.parameter import Parameter
from utils.utils_general import _cuda, sequence_mask, to_onehot
from models.layers import Attention, RNNEncoder, HRNNEncoder, GCNEncoder


class ContextEncoder(nn.Module):
    """
    A GRU recurrent neural network encoder.
    """
    def __init__(self,
                 input_size,
                 embedding_dim,
                 hidden_size,
                 dropout=0.0):
        super(ContextEncoder, self).__init__()
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedder = nn.Embedding(input_size, embedding_dim, padding_idx=PAD_token)
        self.embedder.weight.data.normal_(0, 0.1)

        self.projector = nn.Linear(2*embedding_dim, hidden_size)

        self.rnn = RNNEncoder(
                input_size=embedding_dim,
                hidden_size=embedding_dim*2,
                embedder=None,
                num_layers=1,
                bidirectional=True,
                dropout=dropout)

    def forward(self, inputs, lengths, mask):
        embed = self.embedder(inputs.contiguous())
        embed_dropout = self.dropout_layer(embed)
        context_outputs, context_hidden = self.rnn((embed_dropout, lengths))

        context_outputs = self.projector(context_outputs)
        context_hidden = self.projector(context_hidden)
        return embed, context_outputs, context_hidden


class EntityEncoder(nn.Module):
    def __init__(self,
                 input_size,
                 embedding_dim,
                 hidden_size,
                 embedder,
                 dropout=0.0,
                 B=None):
        super(EntityEncoder, self).__init__()
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedder = embedder
        self.B = B
        
        # For entities in context
        self.relu = nn.ReLU()
        self.mlp1 = nn.Linear(hidden_size*2, hidden_size)

        # For entities in KB
        self.attn = Attention(embedding_dim, hidden_size, embedding_dim, mode='align')
        self.mlp2 = nn.Linear(embedding_dim + hidden_size, hidden_size)

        if self.B == 0:
            self.W = nn.Parameter(torch.empty(self.input_size, self.embedding_dim, self.hidden_size, dtype=torch.float))
            nn.init.xavier_uniform_(self.W)
        else:
            self.W1 = nn.Parameter(torch.empty(self.B, self.embedding_dim, self.hidden_size, dtype=torch.float))
            self.W2 = nn.Parameter(torch.empty(self.input_size, self.B, dtype=torch.float))
            nn.init.xavier_uniform_(self.W1)
            nn.init.xavier_uniform_(self.W2)

        self.W0 = nn.Linear(self.embedding_dim, self.hidden_size, bias=False)
        
    def forward(self, context_emb, context_outputs, context_lengths, context_mask, \
                context_entity_pos, context_entity, context_entity_lengths, context_entity_mask, \
                entity, entity_lengths, entity_mask, \
                kb_entity, kb_entity_lengths, kb_entity_mask, kb_entity_col, kb_entity_nei, indicator):
        batch_size, max_context_len = context_outputs.size(0), context_outputs.size(1)
        entity_set_length = entity.size(1)

        # Processing entities in context
        context_emb = self.dropout_layer(context_emb)
        context_entity_onehot = to_onehot(context_entity_pos, max_context_len, mask=context_entity_mask).float()
        context_hidden = torch.cat((context_emb, context_outputs), dim=2)
        context_entity_hidden = torch.bmm(context_entity_onehot, context_hidden)
        context_entity_hidden = self.relu(self.mlp1(context_entity_hidden)) # B x maxR x h
        #context_entity_hidden = torch.bmm(context_entity_onehot, context_outputs)
        #context_entity_hidden = self.relu(self.mlp1(context_entity_hidden)) # B x maxR x h

        # Processing entities in KB
        entity_emb = self.embedder(entity) # B x maxE x h
        entity_emb = self.dropout_layer(entity_emb)
        entity_align = self.attn(entity_emb, context_emb, mask = context_mask)
        #entity_hidden = torch.cat((entity_emb, entity_align, indicator.unsqueeze(-1).float()), dim=-1) # B x (maxR * maxC) x (2 * h + 1)
        entity_hidden = torch.cat((entity_emb, entity_align), dim=-1) # B x (maxR * maxC) x (2 * h)
        entity_hidden = self.relu(self.mlp2(entity_hidden)) # B x (maxR * maxC) x h

        kb_entity_onehot = to_onehot(kb_entity, entity_set_length, mask=kb_entity_mask).float()
        kb_entity_init = torch.bmm(kb_entity_onehot, entity_hidden) # B x (maxR * maxC) x h

        if self.B == 0:
            W = self.W
        else:
            W = torch.matmul(self.W2, self.W1.permute(1, 0, 2)).permute(1, 0, 2)

        kb_entity_col_hidden = W.index_select(0, kb_entity_col.view(-1))
        kb_entity_col_hidden = kb_entity_col_hidden.contiguous().view(batch_size, -1, \
                                                            self.hidden_size, self.hidden_size) # B x (maxR * maxC) x h2 x h1
        kb_entity_col_hidden = self.dropout_layer(kb_entity_col_hidden)

        kb_entity_state = torch.matmul(kb_entity_col_hidden, kb_entity_init.unsqueeze(3)).squeeze(3)
        kb_entity_right = torch.bmm(kb_entity_nei.float(), kb_entity_state)
        kb_entity_nei_sum = kb_entity_nei.sum(2, keepdim=True, dtype=torch.float)
        kb_entity_nei_sum = torch.clamp(kb_entity_nei_sum, min=1)
        kb_entity_right = kb_entity_right / kb_entity_nei_sum

        kb_entity_left = self.W0(kb_entity_init)
        kb_entity_hidden = self.relu(kb_entity_left + kb_entity_right)

        return context_entity_hidden, kb_entity_hidden


class Decoder(nn.Module):
    def __init__(self, 
                 embedder,
                 vocab,
                 hidden_size,
                 dropout):
        super(Decoder, self).__init__()
        self.num_vocab = vocab.n_words
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout) 

        if embedder:
            self.embedder = embedder
        else:
            self.embedder = nn.Embedding(vocab.n_words, hidden_size, padding_idx=PAD_token)
            self.embedder.weight.data.normal_(0, 0.1)

        self.softmax = nn.Softmax(dim=1)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # For context encoder
        self.projector = nn.Linear(hidden_size, hidden_size)

        # For entity encoder
        self.context_entity_attention = Attention(hidden_size, hidden_size, hidden_size, mode ="general")
        self.kb_entity_attention = Attention(hidden_size, hidden_size, hidden_size, mode ="general")
        self.switch = nn.Linear(hidden_size, 1)

        # For vocabulary
        self.concat = nn.Linear(hidden_size*3, hidden_size)
        self.vocab_matrix =nn.Linear(hidden_size, vocab.n_words)
        self.context_attention = Attention(hidden_size, hidden_size, hidden_size, mode='mlp')

    def forward(self, context_hidden, context_outputs, context_lengths, context_mask, \
                context_entity, context_entity_lengths, context_entity_mask, context_entity_id, \
                kb_entity, kb_entity_id, kb_entity_row, kb_entity_lengths, kb_entity_mask, \
                entity, entity_lengths, entity_mask, entity_plain, entity_type, \
                target_batches, max_target_length, schedule_sampling, get_decoded_words):
        
        batch_size, entity_set_length = entity.size(0), entity.size(1)
        #context_entity_id = context_entity_id + context_entity_mask.long() * (entity_set_length-1)
        #kb_entity_id = kb_entity_id + kb_entity_mask.long() * (entity_set_length-1)

        # Initialize variables for vocab and pointer
        all_decoder_outputs_vocab = _cuda(torch.zeros(max_target_length, batch_size, self.num_vocab))
        all_decoder_outputs_ptr = _cuda(torch.zeros(max_target_length, batch_size, entity_set_length))

        decoder_input = _cuda(torch.LongTensor([SOS_token] * batch_size))
        memory_mask_for_step = _cuda(torch.ones(batch_size, entity_set_length))
        decoded_fine, decoded_coarse = [], []

        #dec_hidden = self.relu(self.projector(context_hidden))
        dec_hidden = self.tanh(self.projector(context_hidden))
        
        # Start to generate word-by-word
        for t in range(max_target_length):
            pre_emb = self.dropout_layer(self.embedder(decoder_input)) # b * e
            if len(pre_emb.size()) == 1: pre_emb = pre_emb.unsqueeze(0)
            
            _, dec_hidden = self.gru(pre_emb.unsqueeze(0), dec_hidden)
            
            # For context distribution
            p_entity_context = _cuda(torch.zeros(batch_size, entity_set_length))

            context_entity_hidden, context_entity_pro = self.context_entity_attention(dec_hidden.transpose(0,1), \
                                                            context_entity, mask=context_entity_mask, return_weights=True)
            p_entity_context.scatter_add_(1, context_entity_id, context_entity_pro.squeeze(1))

            # For KB distribution
            p_entity_kb = _cuda(torch.zeros(batch_size, entity_set_length))
            
            ## Row-level
            kb_entity_row_onehot = to_onehot(kb_entity_row, mask=kb_entity_mask).transpose(1,2) # B x maxR x maxE
            kb_entity_row_hidden = torch.bmm(kb_entity_row_onehot, kb_entity) # B x maxR x h
            kb_entity_row_sum = kb_entity_row_onehot.sum(2, keepdim=True, dtype=torch.float) # B x maxR x 1
            kb_entity_row_mask = kb_entity_row_sum.squeeze(2).eq(0)
            kb_entity_row_sum = torch.clamp(kb_entity_row_sum, min=1)
            kb_entity_row_hidden = kb_entity_row_hidden / kb_entity_row_sum
            kb_entity_hidden, kb_entity_row_pro = self.kb_entity_attention(dec_hidden.transpose(0,1), \
                                                kb_entity_row_hidden, mask=kb_entity_row_mask, return_weights=True)
            kb_entity_row_pro = torch.bmm(kb_entity_row_pro, kb_entity_row_onehot).squeeze(1) # B x maxE

            ## Entity-level
            kb_entity_logit = self.kb_entity_attention(dec_hidden.transpose(0,1), \
                                                            kb_entity, return_weights_only=True) # B x maxE x 1
            kb_entity_logit = kb_entity_logit * kb_entity_row_onehot # B x maxR x maxE
            kb_entity_logit.masked_fill_(torch.logical_not(kb_entity_row_onehot.bool()), -1e9)
            #kb_entity_logit.masked_fill_(1 - kb_entity_row_onehot.byte(), -1e9)
            #kb_entity_logit = kb_entity_logit - (1 - kb_entity_row_onehot) * 1e10
            kb_entity_pro = F.softmax(kb_entity_logit, dim=2)
            kb_entity_pro = torch.gather(kb_entity_pro, 1, kb_entity_row.unsqueeze(1)).squeeze(1)
            #kb_entity_pro = kb_entity_pro.sum(1)
            kb_entity_pro = kb_entity_pro * kb_entity_row_pro
            p_entity_kb.scatter_add_(1, kb_entity_id, kb_entity_pro)

            """
            kb_entity_hidden, kb_entity_logit = self.kb_entity_attention(dec_hidden.transpose(0,1), \
                                                            kb_entity, mask=kb_entity_mask, return_weights=True)
            kb_entity_logit = kb_entity_logit.squeeze(1)
            p_entity_kb.scatter_add_(1, kb_entity_id, kb_entity_logit)
            """

            switch_input = self.switch(dec_hidden.squeeze(0))
            #pro_switch = self.softmax(switch_input)

            #if not get_decoded_words:
            #    pro_switch = nn.functional.gumbel_softmax(switch_input, tau=1.0 - (epoch / 15.0), hard=False)
            #else:
            #    pro_switch = nn.functional.gumbel_softmax(switch_input, tau=1.0 - (epoch / 15.0), hard=True)

            #p_entity = torch.cat((p_entity_context.unsqueeze(2), p_entity_kb.unsqueeze(2)), dim=2)
            #p_entity = torch.bmm(p_entity, pro_switch.unsqueeze(2)).squeeze(2)
            pro_switch = self.sigmoid(switch_input)
            p_entity = (1.0 - pro_switch) * p_entity_context + pro_switch * p_entity_kb

            # For Vocab
            vocab_attn = self.context_attention(dec_hidden.transpose(0,1), context_outputs, mask=context_mask)
            #entity_hidden = torch.cat((context_entity_hidden, kb_entity_hidden), dim=1)
            #entity_hidden = torch.bmm(pro_switch.unsqueeze(1), entity_hidden)
            entity_hidden = context_entity_hidden.squeeze(1) * (1 - pro_switch) + kb_entity_hidden.squeeze(1) * pro_switch
            #concat_input = torch.cat((dec_hidden.squeeze(0), vocab_attn.squeeze(1)), dim=1)
            concat_input = torch.cat((dec_hidden.squeeze(0), vocab_attn.squeeze(1), entity_hidden.squeeze(1)), dim=1)
            concat_output = torch.tanh(self.concat(concat_input))
            #p_vocab = self.attend_vocab(self.embedder.weight, concat_output)
            p_vocab = self.vocab_matrix(concat_output)

            all_decoder_outputs_vocab[t] = p_vocab
            all_decoder_outputs_ptr[t] = p_entity

            use_teacher_forcing = random.random() < schedule_sampling
            if use_teacher_forcing:
                decoder_input = target_batches[:,t] 
            else:
                _, topvi = p_vocab.data.topk(1)
                decoder_input = topvi.squeeze()
            
            if get_decoded_words:
                prob_soft = self.softmax(p_entity)
                search_len = min(5, min(entity_lengths))
                prob_soft = prob_soft * memory_mask_for_step
                _, toppi = prob_soft.data.topk(search_len)
                temp_f, temp_c = [], []
                
                for bi in range(batch_size):
                    token = topvi[bi].item() #topvi[:,0][bi].item()
                    temp_c.append(self.vocab.index2word[token])
                    
                    if '@' in self.vocab.index2word[token]:
                        slot = self.vocab.index2word[token]
                        cw = 'UNK'
                        for i in range(search_len):
                            top_index = toppi[:,i][bi].item()
                            #if top_index < entity_lengths[bi]-1 and entity_type[bi][top_index] == slot:
                            if top_index < entity_lengths[bi]-1:
                                cw = entity_plain[bi][toppi[:,i][bi].item()]
                                break
                        temp_f.append(cw)
                        
                        if args['record']:
                            memory_mask_for_step[bi, toppi[:,i][bi].item()] = 0
                    else:
                        temp_f.append(self.vocab.index2word[token])

                decoded_fine.append(temp_f)
                decoded_coarse.append(temp_c)

        return all_decoder_outputs_vocab, all_decoder_outputs_ptr, decoded_fine, decoded_coarse

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1,0))
        # scores = F.softmax(scores_, dim=1)
        return scores_

