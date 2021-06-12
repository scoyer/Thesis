import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
import random
import numpy as np
import os
import json

from utils.measures import wer, moses_multi_bleu
from utils.masked_cross_entropy import *
from utils.config import *
from models.modules import *


class EER(nn.Module):
    def __init__(self, 
                 task, 
                 vocab,
                 vocab_attribute,
                 max_resp_len, 
                 embedding_dim, 
                 hidden_size, 
                 lr,
                 dropout,
                 path,
                 B,
                 share_embedding):
        super(EER, self).__init__()
        self.name = "Model"
        self.task = task
        self.vocab = vocab
        self.vocab_attribute = vocab_attribute
        self.max_resp_len = max_resp_len
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.input_size = vocab.n_words
        self.output_size = vocab.n_words
        self.attribute_size = vocab_attribute.n_words
        self.lr = lr
        self.dropout = dropout
        self.max_resp_len = max_resp_len
        self.B = B
        self.share_embedding = share_embedding
        self.best = ""

        if path:
            if USE_CUDA:
                print("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path)+'/enc.th')
                self.extKnow = torch.load(str(path)+'/enc_kb.th')
                self.decoder = torch.load(str(path)+'/dec.th')
            else:
                print("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path)+'/enc.th',lambda storage, loc: storage)
                self.extKnow = torch.load(str(path)+'/enc_kb.th',lambda storage, loc: storage)
                self.decoder = torch.load(str(path)+'/dec.th',lambda storage, loc: storage)
        else:
            self.encoder = ContextEncoder(self.input_size, self.embedding_dim, self.hidden_size, dropout)
            self.extKnow = EntityEncoder(self.attribute_size, self.embedding_dim, self.hidden_size, self.encoder.embedder, dropout,B)
            self.decoder = Decoder(self.encoder.embedder if share_embedding else None, vocab, hidden_size, dropout)

        # Initialize optimizers and criterion
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.extKnow_optimizer = optim.Adam(self.extKnow.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer, mode='max', factor=0.5, patience=1, min_lr=0.00005, verbose=True)
        self.reset()

        if USE_CUDA:
            self.encoder.cuda()
            self.extKnow.cuda()
            self.decoder.cuda()

    def print_loss(self):    
        print_loss_avg = self.loss / self.print_every
        print_loss_v = self.loss_v / self.print_every
        print_loss_l = self.loss_l / self.print_every
        self.print_every += 1     
        return 'L:{:.2f},LE:{:.2f},LP:{:.2f}'.format(print_loss_avg, print_loss_v, print_loss_l)
    
    def save_model(self, acc):
        name_data = "KVR/" if self.task=='kvr' else "CAM/"
        oov_test = 'OT' if args["OOVTest"] else ''
        directory = 'save/EER-' \
                    +oov_test \
                    +'DS'+self.task \
                    +'HDD'+str(self.hidden_size) \
                    +'BSZ'+str(args['batch']) \
                    +'DR'+str(self.dropout) \
                    +'lr'+str(self.lr) \
                    +'RS'+(str(self.B) if self.B is not None else str(self.relation_size)) \
                    +str(acc)

        if not os.path.exists(directory):
            os.makedirs(directory)
        
        torch.save(self.encoder, directory + '/enc.th')
        torch.save(self.extKnow, directory + '/enc_kb.th')
        torch.save(self.decoder, directory + '/dec.th')

        self.best = directory

    def reset(self):
        self.loss, self.print_every, self.loss_v, self.loss_l = 0, 1, 0, 0
    
    def _cuda(self, x):
        if USE_CUDA:
            return torch.Tensor(x).cuda()
        else:
            return torch.Tensor(x)

    def train_batch(self, data, clip, reset=0, ss=1.0):
        if reset: self.reset()
        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.extKnow_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        
        # Encode and Decode
        max_target_length = max(data['response_lengths'])
        all_decoder_outputs_vocab, all_decoder_outputs_ptr, _, _ = self.encode_and_decode(data, max_target_length, ss, False)
        
        # Loss calculation and backpropagation
        loss_v = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(), 
            data['sketch_response'].contiguous(), 
            data['response_lengths'])
        loss_l = masked_cross_entropy(
            all_decoder_outputs_ptr.transpose(0, 1).contiguous(), 
            data['response_entity_id'].contiguous(), 
            #data['response_lengths'])
            data['response_lengths'], is_logit=False)
        loss = loss_v + loss_l
        loss.backward()

        # Clip gradient norms
        ec = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        ec = torch.nn.utils.clip_grad_norm_(self.extKnow.parameters(), clip)
        dc = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)

        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.extKnow_optimizer.step()
        self.decoder_optimizer.step()

        self.loss += loss.item()
        self.loss_v += loss_v.item()
        self.loss_l += loss_l.item()

    def encode_and_decode(self, data, max_target_length, schedule_sampling, get_decoded_words):
        # Encode dialog history to vectors
        context_emb, context_outputs, context_hidden = self.encoder(
                                                     data['context'], 
                                                     data['context_lengths'],
                                                     data['context_mask'])

        # Encode KB information to vectors
        context_entity, kb_entity = self.extKnow(context_emb,
                                                     context_outputs,
                                                     data['context_lengths'],
                                                     data['context_mask'],
                                                     data['context_entity_pos'],
                                                     data['context_entity_id'],
                                                     data['context_entity_lengths'],
                                                     data['context_entity_mask'],
                                                     data['entity'],
                                                     data['entity_lengths'],
                                                     data['entity_mask'],
                                                     data['kb_entity_id'],
                                                     data['kb_entity_lengths'],
                                                     data['kb_entity_mask'],
                                                     data['kb_entity_col'],
                                                     data['kb_entity_nei'],
                                                     data['indicator'])
        
        # Decode the dialog history and KB
        outputs_vocab, outputs_ptr, decoded_fine, decoded_coarse = self.decoder(
                                                     context_hidden,
                                                     context_outputs,
                                                     data['context_lengths'],
                                                     data['context_mask'],
                                                     context_entity,
                                                     data['context_entity_lengths'],
                                                     data['context_entity_mask'],
                                                     data['context_entity_id'],
                                                     kb_entity,
                                                     data['kb_entity_id'],
                                                     data['kb_entity_row'],
                                                     data['kb_entity_lengths'],
                                                     data['kb_entity_mask'],
                                                     data['entity'],
                                                     data['entity_lengths'],
                                                     data['entity_mask'],
                                                     data['entity_plain'],
                                                     data['entity_type'],
                                                     data['sketch_response'],
                                                     max_target_length,
                                                     schedule_sampling,
                                                     get_decoded_words)

        return outputs_vocab, outputs_ptr, decoded_fine, decoded_coarse

    def evaluate(self, dev, matric_best, early_stop=None):
        print("STARTING EVALUATION")
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.extKnow.train(False)
        self.decoder.train(False)
        
        ref, hyp = [], []
        dialog_acc_dict = {}
        F1_pred, F1_cal_pred, F1_nav_pred, F1_wet_pred = 0, 0, 0, 0
        F1_count, F1_cal_count, F1_nav_count, F1_wet_count = 0, 0, 0, 0
        pbar = tqdm(enumerate(dev),total=len(dev))
        new_precision, new_recall, new_f1_score = 0, 0, 0

        if self.task == 'kvr':
            with open('data/KVR/kvret_entities.json') as f:
            #with open('data/KVR_2/kvret_entities.json.oov') as f:
                global_entity = json.load(f)
                global_entity_list = []
                for key in global_entity.keys():
                    if key != 'poi':
                        global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]
                    else:
                        for item in global_entity['poi']:
                            global_entity_list += [item[k].lower().replace(' ', '_') for k in item.keys()]
                global_entity_list = list(set(global_entity_list))
        else:
            with open('data/CamRest/camrest_entities.json') as f:
            #with open('data/CamRest_2/camrest_entities.json.oov') as f:
                global_entity_list = json.load(f)

        for j, data_dev in pbar: 
            # Encode and Decode
            _, _, decoded_fine, decoded_coarse = self.encode_and_decode(data_dev, self.max_resp_len, -1.0, True)
            decoded_coarse = np.transpose(decoded_coarse)
            decoded_fine = np.transpose(decoded_fine)
            for bi, row in enumerate(decoded_fine):
                st = ''
                for e in row:
                    if e == 'EOS': break
                    else: st += e + ' '
                st_c = ''
                for e in decoded_coarse[bi]:
                    if e == 'EOS': break
                    else: st_c += e + ' '
                pred_sent = st.lstrip().rstrip()
                pred_sent_coarse = st_c.lstrip().rstrip()
                gold_sent = data_dev['response_plain'][bi].lstrip().rstrip()
                ref.append(gold_sent)
                hyp.append(pred_sent)
                
                if self.task == 'kvr': 
                    # compute F1 SCORE
                    single_f1, count = self.compute_prf(data_dev['ent_index'][bi], pred_sent.split(), global_entity_list, data_dev['entity_plain'][bi])
                    F1_pred += single_f1
                    F1_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_cal'][bi], pred_sent.split(), global_entity_list, data_dev['entity_plain'][bi])
                    F1_cal_pred += single_f1
                    F1_cal_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_nav'][bi], pred_sent.split(), global_entity_list, data_dev['entity_plain'][bi])
                    F1_nav_pred += single_f1
                    F1_nav_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_wet'][bi], pred_sent.split(), global_entity_list, data_dev['entity_plain'][bi])
                    F1_wet_pred += single_f1
                    F1_wet_count += count
                elif self.task == 'cam':
                    single_f1, count = self.compute_prf(data_dev['ent_index'][bi], pred_sent.split(), [], data_dev['entity_plain'][bi])
                    F1_pred += single_f1
                    F1_count += count

                if args['genSample']:
                    self.print_examples(bi, data_dev, pred_sent, pred_sent_coarse, gold_sent)

        # Set back to training mode
        self.encoder.train(True)
        self.extKnow.train(True)
        self.decoder.train(True)

        bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref), lowercase=True)
        F1_score = F1_pred / float(F1_count)

        if self.task == 'kvr':
            print("F1 SCORE:\t{}".format(F1_pred/float(F1_count)))
            print("\tCAL F1:\t{}".format(F1_cal_pred/float(F1_cal_count))) 
            print("\tWET F1:\t{}".format(F1_wet_pred/float(F1_wet_count))) 
            print("\tNAV F1:\t{}".format(F1_nav_pred/float(F1_nav_count))) 
            print("BLEU SCORE:\t"+str(bleu_score))
        elif self.task == 'cam':
            print("F1 SCORE:\t{}".format(F1_pred/float(F1_count)))
            print("BLEU SCORE:\t"+str(bleu_score))
        
        if (early_stop == 'BLEU'):
            if (bleu_score >= matric_best):
                self.save_model('BLEU-'+str(bleu_score))
                print("MODEL SAVED")
            return bleu_score
        elif (early_stop == 'ENTF1'):
            if (F1_score >= matric_best):
                self.save_model('ENTF1-{:.4f}'.format(F1_score))
                print("MODEL SAVED")  
            return F1_score

    def compute_prf(self, gold, pred, global_entity_list, kb_plain):
        local_kb_word = kb_plain[1:-1]
        TP, FP, FN = 0, 0, 0
        if len(gold)!= 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in set(pred):
                if p in global_entity_list or p in local_kb_word:
                    if p not in gold:
                        FP += 1
            precision = TP / float(TP+FP) if (TP+FP)!=0 else 0
            recall = TP / float(TP+FN) if (TP+FN)!=0 else 0
            F1 = 2 * precision * recall / float(precision + recall) if (precision+recall)!=0 else 0
        else:
            precision, recall, F1, count = 0, 0, 0, 0
        return F1, count

    def print_examples(self, batch_idx, data, pred_sent, pred_sent_coarse, gold_sent):
        conv_plain = data['context_arr_plain'][batch_idx][-1]

        print('User Query:', " ".join(conv_plain))
        print('Sketch System Response : ', pred_sent_coarse)
        print('Final System Response : ', pred_sent)
        print('Gold System Response : ', gold_sent)
        print('\n')
