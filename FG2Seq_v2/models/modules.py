import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from utils.config import *
from torch.nn.parameter import Parameter
from utils.utils_general import _cuda, mask_and_length, to_onehot
from models.layers import SelfAttention, Attention, RNNEncoder, HRNNEncoder, RGCNEncoder, GCNEncoder, COMPGCNEncoder


class DualAttentionDecoder(nn.Module):
    def __init__(self, embedder, lang, embedding_dim, dropout):
        super(DualAttentionDecoder, self).__init__()
        self.num_vocab = lang.n_words
        self.lang = lang
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout) 

        if embedder:
            self.embedder = embedder
        else:
            self.embedder = nn.Embedding(lang.n_words, embedding_dim, padding_idx=PAD_token)
            self.embedder.weight.data.normal_(0, 0.1)

        self.softmax = nn.Softmax(dim=1)
        self.gru = nn.GRU(embedding_dim, embedding_dim*2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.projector = nn.Linear(2*embedding_dim,embedding_dim*2)
        self.softmax = nn.Softmax(dim = 1)
        self.knowledge_attention = Attention(embedding_dim*2, embedding_dim*2, embedding_dim, mode='mlp')
        self.context_attention = Attention(embedding_dim*2, embedding_dim*2, embedding_dim, mode='mlp')
        self.concat = nn.Linear(6*embedding_dim, embedding_dim)
        self.entity_ranking = Attention(embedding_dim, embedding_dim*2, embedding_dim, mode='mlp', return_attn_only=True)
        
        self.vocab_matrix =nn.Linear(embedding_dim, lang.n_words)

    def forward(self, kb_ent, extKnow, context, context_mask, copy_list, encode_hidden, target_batches, max_target_length, schedule_sampling, get_decoded_words):
        batch_size = len(copy_list)
        story_size = max([len(seq) for seq in copy_list])
        extKnow_mask, _ = mask_and_length(kb_ent, PAD_token)

        # Initialize variables for vocab and pointer
        all_decoder_outputs_vocab = _cuda(torch.zeros(max_target_length, batch_size, self.num_vocab))
        all_decoder_outputs_ptr = _cuda(torch.zeros(max_target_length, batch_size, story_size))
        decoder_input = _cuda(torch.LongTensor([SOS_token] * batch_size))
        decoded_fine, decoded_coarse = [], []
        
        #hidden = self.relu(self.projector(encode_hidden)).unsqueeze(0)
        hidden = self.tanh(self.projector(encode_hidden)).unsqueeze(0)
        
        # Start to generate word-by-word
        for t in range(max_target_length + 1):
            rnn_input_list, concat_input_list = [], []

            embed_q = self.dropout_layer(self.embedder(decoder_input)) # b * e
            if len(embed_q.size()) == 1: embed_q = embed_q.unsqueeze(0)
            rnn_input_list.append(embed_q)

            rnn_input = torch.cat(rnn_input_list, dim=1)
            _, hidden = self.gru(rnn_input.unsqueeze(0), hidden)
            concat_input_list.append(hidden.squeeze(0))

            #get knowledge attention
            knowledge_outputs = self.knowledge_attention(hidden.transpose(0,1), extKnow, mask=extKnow_mask)
            concat_input_list.append(knowledge_outputs.squeeze(1))

            #get context attention
            context_outputs = self.context_attention(hidden.transpose(0,1), context, mask=context_mask)
            concat_input_list.append(context_outputs.squeeze(1))

            #concat_input = torch.cat((hidden.squeeze(0), context_outputs.squeeze(1), knowledge_outputs.squeeze(1)), dim=1)
            concat_input = torch.cat(concat_input_list, dim=1)
            concat_output = torch.tanh(self.concat(concat_input))

            if t < max_target_length:
                #p_vocab = self.attend_vocab(self.C.weight, concat_output)
                p_vocab = self.vocab_matrix(concat_output)
                all_decoder_outputs_vocab[t] = p_vocab

            if t > 0:
                p_entity = self.entity_ranking(concat_output.unsqueeze(1), extKnow, mask=extKnow_mask).squeeze(1)
                all_decoder_outputs_ptr[t - 1] = p_entity

            if t < max_target_length:
                use_teacher_forcing = random.random() < schedule_sampling
                if use_teacher_forcing:
                    decoder_input = target_batches[:,t] 
                else:
                    _, topvi = p_vocab.data.topk(1)
                    decoder_input = topvi.squeeze()

        # Start to generate word-by-word
        if get_decoded_words:
            for t in range(max_target_length):
                p_vocab = all_decoder_outputs_vocab[t]
                p_entity = all_decoder_outputs_ptr[t]
                _, topvi = p_vocab.data.topk(1)

                search_len = min(5, story_size)
                _, toppi = p_entity.data.topk(search_len)
                temp_f, temp_c = [], []
                
                for bi in range(batch_size):
                    token = topvi[bi].item() #topvi[:,0][bi].item()
                    temp_c.append(self.lang.index2word[token])
                    
                    if '@' in self.lang.index2word[token]:
                        cw = 'UNK'
                        for i in range(search_len):
                            #if toppi[:,i][bi] < story_lengths[bi]-1: 
                            if toppi[:,i][bi] > 0 and toppi[:,i][bi] < len(copy_list[bi]): 
                                cw = copy_list[bi][toppi[:,i][bi].item()]            
                                break

                        temp_f.append(cw)
                    else:
                        temp_f.append(self.lang.index2word[token])

                decoded_fine.append(temp_f)
                decoded_coarse.append(temp_c)

        return all_decoder_outputs_vocab, all_decoder_outputs_ptr, decoded_fine, decoded_coarse

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1,0))
        # scores = F.softmax(scores_, dim=1)
        return scores_

class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class ContextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dropout):
        super(ContextEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_token)
        self.embedding.weight.data.normal_(0, 0.1)

        #define two RNNEncoders and one HRNNEncoder
        self.question_rnn1 = RNNEncoder(
                input_size=embedding_dim,
                hidden_size=embedding_dim*2,
                embedder=None,
                num_layers=1,
                bidirectional=True,
                dropout=dropout)
        self.question_rnn2 = RNNEncoder(
                input_size=embedding_dim*2,
                hidden_size=embedding_dim,
                embedder=None,
                num_layers=1,
                bidirectional=False,
                dropout=dropout)
        self.hier_question_rnn = HRNNEncoder(self.question_rnn1, self.question_rnn2)

    def forward(self, x2):
        x2_mask, x2_lengths = mask_and_length(x2, PAD_token)

        x2_embed = self.embedding(x2.contiguous())

        #add dropout
        x2_embed = self.dropout_layer(x2_embed)

        hiera_outputs, hiera_hidden, sub_outputs, sub_hidden, last_sub_outputs, last_sub_lengths = self.hier_question_rnn((x2_embed, x2_lengths), x2_mask)

        # Get the question mask
        question_len = x2_lengths.gt(0).long().sum(dim=1)
        question_mask = torch.stack(
                [x2_mask[b, l - 1] for b, l in enumerate(question_len)])
        max_len = last_sub_lengths.max()
        question_mask = question_mask[:, :max_len]

        return x2_embed, sub_outputs, sub_hidden, hiera_outputs, hiera_hidden, last_sub_outputs, last_sub_lengths, question_mask


class KnowledgeEncoder(nn.Module):
    def __init__(self, embedding, vocab_size, embedding_dim, relation_size, dropout, B):
        super(KnowledgeEncoder, self).__init__()
        #Embedding parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        self.vocab_size = vocab_size
        self.relation_size = relation_size

        if embedding:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_token)
            self.embedding.weight.data.normal_(0, 0.1)
        
        self.relu = nn.ReLU()
        self.B = B
        
        #get C_i_1
        self.question_attn1 = Attention(embedding_dim, embedding_dim, embedding_dim, mode='mlp')
        #self.mlp1 = nn.Sequential(
        #    nn.Linear(embedding_dim*2 + 1, embedding_dim),
        #    nn.ReLU(),
        #)
        #self.dialog_flow1 = RNNEncoder(embedding_dim, embedding_dim, embedder=None, num_layers=1, bidirectional=False)
        self.dialog_flow1 = RNNEncoder(embedding_dim * 2 + 1, embedding_dim, embedder=None, num_layers=1, bidirectional=False)
        # self.gcn1 = RGCNEncoder(embedding_dim, embedding_dim, self.relation_size, self.B, dropout)
        self.gcn1 = GCNEncoder(embedding_dim, embedding_dim, self.relation_size, dropout, B=self.B)
        #self.gcn1 = COMPGCNEncoder(embedding_dim, embedding_dim, self.relation_size, dropout)

        #get C_i_2
        self.question_attn2 = Attention(embedding_dim * 2, embedding_dim * 2, embedding_dim, mode='mlp')
        #self.mlp2 = nn.Sequential(
        #    nn.Linear(embedding_dim*4 + 1, embedding_dim),
        #    nn.ReLU(),
        #)
        #self.dialog_flow2 = RNNEncoder(embedding_dim, embedding_dim, embedder=None, num_layers=1, bidirectional=False)
        self.dialog_flow2 = RNNEncoder(embedding_dim * 4 + 1, embedding_dim, embedder=None, num_layers=1, bidirectional=False)
        # self.gcn2 = RGCNEncoder(embedding_dim, embedding_dim, self.relation_size, self.B, dropout)
        self.gcn2 = GCNEncoder(embedding_dim, embedding_dim, self.relation_size, dropout, B=self.B)
        #self.gcn2 = COMPGCNEncoder(embedding_dim, embedding_dim, self.relation_size, dropout)

        #self-attention
        self.entity_attention = SelfAttention(embedding_dim, embedding_dim)

    def graph_norm(self, graph):
        graph = graph.to_dense()
        batch_size = graph.size(0)
        
        degree = torch.sum(graph, dim=-1, keepdim=True).clamp(min=1)
        graph = graph / degree  
        return graph

    def forward(self, context, context_embed, context_outputs, kb_ent, kb_rel, kb_obj, kb_f, kb_m, graph):
        len_b, len_n, len_l = context.size()
        len_k = kb_ent.size(1)

        context_mask, context_lengths = mask_and_length(context, PAD_token)
        context_outputs = context_outputs.contiguous().view(len_b, len_n, len_l, -1)
        
        kb_mask, kb_lengths = mask_and_length(kb_ent, PAD_token)

        # first Flow-to-Graph
        kb_embed = self.embedding(kb_ent)
        kb_embed = self.dropout_layer(kb_embed)

        kb_embed_expand = kb_embed.unsqueeze(1).expand(-1, len_n, -1, -1).contiguous().view(len_b*len_n, len_k, -1) #(b * q_num) * len_k * em1
        kb_mask_expand = kb_mask.unsqueeze(1).expand(-1, len_n, -1).contiguous().view(len_b*len_n, len_k) #(b * q_num) * len_k

        graph = self.graph_norm(graph)
        graph_expand = graph.unsqueeze(1).expand(graph.size(0), len_n, graph.size(1), graph.size(2), graph.size(3))
        graph_expand = graph_expand.contiguous().view(-1, graph.size(1), graph.size(2), graph.size(3))

        kb_rel_expand = kb_rel.unsqueeze(1).expand(-1, len_n, -1, -1).contiguous().view(len_b*len_n, len_k, -1)
        kb_obj_expand = kb_obj.unsqueeze(1).expand(-1, len_n, -1, -1).contiguous().view(len_b*len_n, len_k, -1)
        kb_m_expand = kb_m.unsqueeze(1).expand(-1, len_n, -1, -1).contiguous().view(len_b*len_n, len_k, -1)

        context_embed = context_embed.contiguous().view(len_b*len_n, len_l, -1)
        context_mask = context_mask.view(len_b*len_n, len_l)

        kb_attn1 = self.question_attn1(kb_embed_expand, context_embed, mask=context_mask) #(b * q_num) * len_k * em2
        flow_input_1 = torch.cat([kb_embed_expand, kb_attn1, kb_f.view(len_b*len_n, len_k, 1)], dim=2) #(b * q_num) * len_k * (em1 + em2 + n_feat)
        #flow_input_1 = self.mlp1(flow_input_1)

        # reshape
        flow_input_1 = flow_input_1.view(len_b, len_n, len_k, -1).transpose(1, 2)
        flow_input_1 = flow_input_1.contiguous().view(len_b * len_k, len_n, -1)

        flow_output_1,_ = self.dialog_flow1(flow_input_1)

        # reshape
        flow_output_1 = flow_output_1.view(len_b, len_k, len_n, -1).transpose(1, 2)
        flow_output_1 = flow_output_1.contiguous().view(len_b * len_n, len_k, -1)

        # C1_2 = self.gcn1(C1_1, kb_rel_expand, kb_obj_expand)
        graph_output_1 = self.gcn1(flow_output_1, graph_expand)
        #graph_output_1 = self.gcn1(flow_output_1, kb_rel_expand, kb_obj_expand, kb_m_expand)

        # second Flow-to-Graph
        context_outputs = context_outputs.contiguous().view(len_b * len_n, len_l, -1)

        kb_output_1 = torch.cat((flow_output_1, graph_output_1), dim=2)

        kb_attn2 = self.question_attn2(kb_output_1, context_outputs, mask=context_mask)
        #flow_input_2 = torch.cat((kb_output_1, kb_attn2), dim=2)
        flow_input_2 = torch.cat([kb_output_1, kb_attn2, kb_f.view(len_b*len_n, len_k, 1)], dim=2)
        #flow_input_2 = self.mlp2(flow_input_2)

        # reshape
        flow_input_2 = flow_input_2.view(len_b, len_n, len_k, -1).transpose(1, 2)
        flow_input_2 = flow_input_2.contiguous().view(len_b * len_k, len_n, -1)

        sentence_lengths = context_lengths.gt(0).long().sum(dim=1)
        sentence_lengths_expand = sentence_lengths.unsqueeze(1).expand(-1, len_k)
        sentence_lengths_expand = sentence_lengths_expand.contiguous().view(len_b * len_k)

        flow_inputs = (flow_input_2, sentence_lengths_expand)
        _, flow_output_2 = self.dialog_flow2(flow_inputs)

        # reshape
        flow_output_2 = flow_output_2.contiguous().view(len_b, len_k, -1)

        # C2_2 = self.gcn2(C2_1, kb_rel_expand, kb_obj_expand)
        graph_output_2 = self.gcn2(flow_output_2, graph)
        #graph_output_2 = self.gcn1(flow_output_2, kb_rel, kb_obj, kb_m)

        knowledge_outputs = torch.cat((flow_output_2, graph_output_2), dim=2)
        #knowledge_hidden = self.entity_attention(knowledge_outputs, x_mask = kb_mask).unsqueeze(1)
        #knowledge_hidden = self.entity_attention(graph_output_2, x_mask = kb_mask).unsqueeze(1)
        knowledge_hidden = graph_output_2[:,0].unsqueeze(1)
        
        return knowledge_outputs, knowledge_hidden
        
