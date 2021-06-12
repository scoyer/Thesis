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

from utils.eval import get_result


class FG2Seq(nn.Module):
    def __init__(self, hidden_size, lang, max_resp_len, path, task, lr, dropout, relation_size, B, share_embedding):
        super(FG2Seq, self).__init__()
        self.name = "FG2Seq"
        self.task = task
        self.hidden_size = hidden_size
        self.vocabulary = lang[0]
        self.relation = lang[1]
        self.input_size = self.vocabulary.n_words
        self.output_size = self.vocabulary.n_words
        self.lr = lr
        self.dropout = dropout
        self.max_resp_len = max_resp_len
        self.relation_size = relation_size
        self.B = B
        self.share_embedding = share_embedding

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
            self.encoder = ContextEncoder(self.vocabulary.n_words, hidden_size, dropout)
            self.extKnow = KnowledgeEncoder(self.encoder.embedding if share_embedding else None, self.vocabulary.n_words, hidden_size, relation_size, dropout, B)
            self.decoder = DualAttentionDecoder(self.encoder.embedding if share_embedding else None, self.vocabulary, hidden_size, dropout) #Generator(lang, hidden_size, dropout)
        
        # Initialize optimizers and criterion
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.extKnow_optimizer = optim.Adam(self.extKnow.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer, mode='max', factor=0.5, patience=1, min_lr=0.0001, verbose=True)
        self.reset()
        
        if USE_CUDA:
            self.encoder.cuda()
            self.extKnow.cuda()
            self.decoder.cuda()

        self.best = None

    def print_loss(self):    
        print_loss_avg = self.loss / self.print_every
        print_loss_v = self.loss_v / self.print_every
        print_loss_l = self.loss_l / self.print_every
        self.print_every += 1     
        return 'L:{:.2f},LE:{:.2f},LP:{:.2f}'.format(print_loss_avg, print_loss_v, print_loss_l)
    
    def save_model(self, acc):
        name_data = "KVR/" if self.task=='kvr' else "CAM/"
        directory = 'save/FG2Seq-' \
                    +args["addName"] \
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

    #def train_batch(self, optimizer, data, clip, reset=0, ss=1.0):
    def train_batch(self, data, clip, reset=0, ss=1.0):
        if reset: self.reset()
        
        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.extKnow_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        """
        optimizer.zero_grad()
        """
        """
        for k, v in data.items():
            print(k)
            if isinstance(v, torch.Tensor):
                print(v.size())
                print(v)
            else:
                print(v)
        """
        # Encode and Decode
        max_target_length = data["sketch_response"].size(1)
        all_decoder_outputs_vocab, all_decoder_outputs_ptr, _, _ = self.encode_and_decode(data, max_target_length, ss, False)
        
        # Loss calculation and backpropagation
        loss_v = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(), 
            data['sketch_response'].contiguous(), 
            data['response'], PAD_token)
        loss_l = masked_cross_entropy(
            all_decoder_outputs_ptr.transpose(0, 1).contiguous(), 
            data['ptr_index'].contiguous(), 
            data['response'], PAD_token)
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
        """
        torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        optimizer.step()
        """
        self.loss += loss.item()
        self.loss_v += loss_v.item()
        self.loss_l += loss_l.item()
    
    def encode_and_decode(self, data, max_target_length, schedule_sampling, get_decoded_words):
        # Encode dialog history to vectors
        context_embed, context_outputs, _, _, context_hidden, query_outputs, query_lengths, query_mask = self.encoder(data['context_arr'])

        # Encode KB information to vectors
        knowledge_outputs, knowledge_hidden = self.extKnow(
                                                    data["context_arr"],
                                                    context_embed,
                                                    context_outputs,
                                                    data['kb_ent'],
                                                    data['kb_rel'],
                                                    data['kb_obj'],
                                                    data['kb_f'],
                                                    data['kb_m'],
                                                    data['graph'])

        encoded_hidden = torch.cat((context_hidden.squeeze(0), knowledge_hidden.squeeze(1)), dim=1)

        outputs_vocab, outputs_ptr, decoded_fine, decoded_coarse = self.decoder.forward(
            data['kb_ent'],
            knowledge_outputs, 
            query_outputs,
            query_mask,
            data['entity_list'], 
            encoded_hidden,
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
        
        dialog = []
        pbar = tqdm(enumerate(dev),total=len(dev))

        for j, data_dev in pbar: 
            # Encode and Decode
            with torch.no_grad():
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
                gold_sent = ' '.join(data_dev['response'][bi])
                gold_sent = gold_sent.lstrip().rstrip()

                dialog.append({
                        "kb": data_dev["entity_list"][bi],
                        "gold_entity": data_dev["ent_index"][bi],
                        "task": data_dev["domain"][bi] if "domain" in data_dev else None,
                        "result": pred_sent,
                        "target": gold_sent,
                        "dialog_id": data_dev["ID"][bi],
                    })
                

                if args['genSample']:
                    out_str = self.print_examples(bi, data_dev, pred_sent, pred_sent_coarse, gold_sent)
                    print(out_str)


        # Set back to training mode
        self.encoder.train(True)
        self.extKnow.train(True)
        self.decoder.train(True)

        bleu_score, F1_score = get_result(self.task, dialog)

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
    
    """
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
        F1_res_pred, F1_hot_pred, F1_att_pred = 0, 0, 0
        F1_res_count, F1_hot_count, F1_att_count = 0, 0, 0
        pbar = tqdm(enumerate(dev),total=len(dev))
        new_precision, new_recall, new_f1_score = 0, 0, 0

        if self.task == 'kvr':
            with open('data/KVR/kvret_entities.json') as f:
                global_entity = json.load(f)
                global_entity_list = []
                for key in global_entity.keys():
                    if key != 'poi':
                        global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]
                    else:
                        for item in global_entity['poi']:
                            global_entity_list += [item[k].lower().replace(' ', '_') for k in item.keys()]
                global_entity_list = list(set(global_entity_list))
        elif self.task == 'woz':
            with open('data/MultiWOZ/global_entities.json') as f:
                global_entity = json.load(f)
                global_entity_list = []
                for key in global_entity.keys():
                    global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]

        for j, data_dev in pbar: 
            # Encode and Decode
            with torch.no_grad():
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
                gold_sent = ' '.join(data_dev['response'][bi])
                gold_sent = gold_sent.lstrip().rstrip()
                ref.append(gold_sent)
                hyp.append(pred_sent)
                
                if self.task == 'kvr': 
                    # compute F1 SCORE
                    single_f1, count = self.compute_prf(data_dev['ent_index'][bi], pred_sent.split(), global_entity_list, data_dev['entity_list'][bi])
                    F1_pred += single_f1
                    F1_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_cal'][bi], pred_sent.split(), global_entity_list, data_dev['entity_list'][bi])
                    F1_cal_pred += single_f1
                    F1_cal_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_nav'][bi], pred_sent.split(), global_entity_list, data_dev['entity_list'][bi])
                    F1_nav_pred += single_f1
                    F1_nav_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_wet'][bi], pred_sent.split(), global_entity_list, data_dev['entity_list'][bi])
                    F1_wet_pred += single_f1
                    F1_wet_count += count
                elif self.task == 'woz': 
                    # compute F1 SCORE
                    single_f1, count = self.compute_prf(data_dev['ent_index'][bi], pred_sent.split(), global_entity_list, data_dev['entity_list'][bi])
                    F1_pred += single_f1
                    F1_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_res'][bi], pred_sent.split(), global_entity_list, data_dev['entity_list'][bi])
                    F1_res_pred += single_f1
                    F1_res_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_hot'][bi], pred_sent.split(), global_entity_list, data_dev['entity_list'][bi])
                    F1_hot_pred += single_f1
                    F1_hot_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_att'][bi], pred_sent.split(), global_entity_list, data_dev['entity_list'][bi])
                    F1_att_pred += single_f1
                    F1_att_count += count
                elif self.task == 'cam':
                    single_f1, count = self.compute_prf(data_dev['ent_index'][bi], pred_sent.split(), [], data_dev['entity_list'][bi])
                    F1_pred += single_f1
                    F1_count += count

                if args['genSample']:
                    self.print_examples(bi, data_dev, pred_sent, pred_sent_coarse, gold_sent)

        # Set back to training mode
        self.encoder.train(True)
        self.extKnow.train(True)
        self.decoder.train(True)

        bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref), lowercase=True)

        if self.task == 'kvr':
            F1_score = F1_pred / float(F1_count)
            print("F1 SCORE:\t{}".format(F1_pred/float(F1_count)))
            print("\tCAL F1:\t{}".format(F1_cal_pred/float(F1_cal_count))) 
            print("\tWET F1:\t{}".format(F1_wet_pred/float(F1_wet_count))) 
            print("\tNAV F1:\t{}".format(F1_nav_pred/float(F1_nav_count))) 
            print("BLEU SCORE:\t"+str(bleu_score))
        elif self.task == 'woz':
            F1_score = F1_pred / float(F1_count)
            print("F1 SCORE:\t{}".format(F1_pred/float(F1_count)))
            print("\tRES F1:\t{}".format(F1_res_pred/float(F1_res_count))) 
            print("\tHOT F1:\t{}".format(F1_hot_pred/float(F1_hot_count))) 
            print("\tATT F1:\t{}".format(F1_att_pred/float(F1_att_count))) 
            print("BLEU SCORE:\t"+str(bleu_score))
        elif self.task == 'cam':
            F1_score = F1_pred / float(F1_count)
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
    """
    def compute_prf(self, gold, pred, global_entity_list, kb_plain):
        local_kb_word = [k for k in kb_plain]
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
        out_str = "-------------------************************-------------------------------------\n"
        out_str += "External Knowledge:"
        out_str += "\n"
        for kb in data["kb_arr"][batch_idx]:
            out_str += "\t".join(kb)
            out_str += "\n"
        out_str += "\n"

        out_str += "Entity List:"
        out_str += "\n"
        out_str += "\t".join(data["entity_list"][batch_idx])
        out_str += "\n"
        out_str += "\n"

        out_str += 'Context:'
        out_str += "\n"
        for i, sent in enumerate(data["ct_arr"][batch_idx]):
            out_str += "USER: " if i % 2 == 0 else "SYS: "
            out_str += "\n"
            out_str += " ".join(sent)
            out_str += "\n"
        out_str += "\n"

        out_str += ('Gold System Response: ' + gold_sent)
        out_str += "\n"
        out_str += ('Sketch System Response: ' + pred_sent_coarse)
        out_str += "\n"
        out_str += ('Final System Response: ' + pred_sent)
        out_str += '\n'
        return out_str
