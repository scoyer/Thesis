# -*- coding: utf-8 -*-
"""
File: eval.py
"""
import json
import numpy as np
from utils.measures import wer, moses_multi_bleu


def compute_prf(gold_entity_list, pred_sent, global_entity_list, local_kb_word):
    """
    compute entity precision/recall/F1 score
    """
    TP, FP, FN = 0, 0, 0
    if len(gold_entity_list) != 0:
        count = 1
        for g in gold_entity_list:
            if g in pred_sent:
                TP += 1
            else:
                FN += 1
        for p in set(pred_sent):
            if p in global_entity_list or p in local_kb_word:
                if p not in gold_entity_list:
                    FP += 1
        F1 = compute_f1(TP, FP, FN)
    else:
        F1, count = 0, 0
    return TP, FP, FN, F1, count


def compute_f1(tp_count, fp_count, fn_count):
    """
    compute f1 score
    """
    precision = tp_count / float(tp_count + fp_count) if (tp_count + fp_count) != 0 else 0
    recall = tp_count / float(tp_count + fn_count) if (tp_count + fn_count) != 0 else 0
    f1_score = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
    return f1_score


def eval_bleu(data):
    hyps = []
    refs = []
    for dialog in data:
        pred_str = dialog["result"]
        gold_str = dialog["target"]
        hyps.append(pred_str)
        refs.append(gold_str)
    assert len(hyps) == len(refs)
    hyp_arrys = np.array(hyps)
    ref_arrys = np.array(refs)

    bleu_score = moses_multi_bleu(hyp_arrys, ref_arrys, lowercase=True)
    return bleu_score


def cal_resp_acc(gold_str, pred_str):
    targets = gold_str.split()
    preds = pred_str.split()
    max_len = max(len(targets), len(preds))
    if len(preds) < max_len:
        pads = ['<PAD>'] * (max_len-len(preds))
        preds += pads
    else:
        pads = ['<PAD>'] * (max_len-len(targets))
        targets += pads

    token_acc_list = [1 if targets[i] == preds[i] else 0 for i in range(max_len)]
    resp_acc = np.mean(token_acc_list)
    return resp_acc


def eval_dialog_accuracy(data):
    dialog_acc_dict = dict()
    resp_acc_list = []
    for dialog in data:
        dialog_id = dialog["dialog_id"]
        pred_str = dialog["result"]
        gold_str = dialog["target"]
        resp_acc = cal_resp_acc(gold_str, pred_str)
        resp_acc_list.append(resp_acc)
        if dialog_id not in dialog_acc_dict.keys():
            dialog_acc_dict[dialog_id] = []
        dialog_acc_dict[dialog_id].append(resp_acc)
    resp_acc_score = np.mean(resp_acc_list)
    dialog_acc_list = [np.mean(dialog_acc_dict[k]) for k in dialog_acc_dict.keys()]
    dialog_acc_score = np.mean(dialog_acc_list)
    return resp_acc_score, dialog_acc_score


def eval_entity_f1_kvr(data, entity_fp, average="micro"):
    test_data = []
    for dialog in data:
        ent_idx_sch, ent_idx_wet, ent_idx_nav = [], [], []
        #if len(dialog["gold_entity"]) > 0:
        #    dialog["gold_entity"] = ' '.join(dialog["gold_entity"]).replace('_', ' ').split()
        if dialog["task"] == "schedule":
            ent_idx_sch = dialog["gold_entity"]
        elif dialog["task"] == "weather":
            ent_idx_wet = dialog["gold_entity"]
        elif dialog["task"] == "navigate":
            ent_idx_nav = dialog["gold_entity"]
        ent_index = list(set(ent_idx_sch + ent_idx_wet + ent_idx_nav))
        dialog["ent_index"] = ent_index
        dialog["ent_idx_sch"] = list(set(ent_idx_sch))
        dialog["ent_idx_wet"] = list(set(ent_idx_wet))
        dialog["ent_idx_nav"] = list(set(ent_idx_nav))
        test_data.append(dialog)

    with open(entity_fp, 'r') as fr:
        global_entity = json.load(fr)
        global_entity_list = []
        for key in global_entity.keys():
            if key != 'poi':
                global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]
            else:
                for item in global_entity['poi']:
                    global_entity_list += [item[k].lower().replace(' ', '_') for k in item.keys()]
    global_entity_list = list(set(global_entity_list))

    F1_pred, F1_sch_pred, F1_nav_pred, F1_wet_pred = 0, 0, 0, 0
    F1_count, F1_sch_count, F1_nav_count, F1_wet_count = 0, 0, 0, 0
    TP_all, FP_all, FN_all = 0, 0, 0
    TP_sch, FP_sch, FN_sch = 0, 0, 0
    TP_wet, FP_wet, FN_wet = 0, 0, 0
    TP_nav, FP_nav, FN_nav = 0, 0, 0
    
    for dialog in test_data:
        #pred_tokens = dialog["result"].replace('_', ' ').split()
        pred_tokens = dialog["result"].split()
        kb_arrys = dialog["kb"]

        gold_ents = dialog["ent_index"]
        tp, fp, fn, f1, count = compute_prf(gold_ents, pred_tokens, global_entity_list, kb_arrys)
        TP_all += tp
        FP_all += fp
        FN_all += fn
        F1_pred += f1
        F1_count += count

        gold_sch_ents = dialog["ent_idx_sch"]
        tp, fp, fn, f1, count = compute_prf(gold_sch_ents, pred_tokens, global_entity_list, kb_arrys)
        TP_sch += tp
        FP_sch += fp
        FN_sch += fn
        F1_sch_pred += f1
        F1_sch_count += count     

        gold_wet_ents = dialog["ent_idx_wet"]
        tp, fp, fn, f1, count = compute_prf(gold_wet_ents, pred_tokens, global_entity_list, kb_arrys)
        TP_wet += tp
        FP_wet += fp
        FN_wet += fn
        F1_wet_pred += f1
        F1_wet_count += count    

        gold_nav_ents = dialog["ent_idx_nav"]
        tp, fp, fn, f1, count = compute_prf(gold_nav_ents, pred_tokens, global_entity_list, kb_arrys)
        TP_nav += tp
        FP_nav += fp
        FN_nav += fn
        F1_nav_pred += f1
        F1_nav_count += count
    
    if average == "micro":
        F1_score = compute_f1(TP_all, FP_all, FN_all)
        F1_sch_score = compute_f1(TP_sch, FP_sch, FN_sch)
        F1_wet_score = compute_f1(TP_wet, FP_wet, FN_wet)
        F1_nav_score = compute_f1(TP_nav, FP_nav, FN_nav)
    else:
        F1_score = F1_pred / float(F1_count)
        F1_sch_score = F1_sch_pred / float(F1_sch_count)
        F1_wet_score = F1_wet_pred / float(F1_wet_count)
        F1_nav_score = F1_nav_pred / float(F1_nav_count)

    return F1_score, F1_sch_score, F1_wet_score, F1_nav_score


def eval_entity_f1_multiwoz(data, entity_fp, average="micro"):
    test_data = []
    for dialog in data:
        ent_idx_res, ent_idx_att, ent_idx_hotel = [], [], []
        #if len(dialog["gold_entity"]) > 0:
        #    dialog["gold_entity"] = ' '.join(dialog["gold_entity"]).replace('_', ' ').split()
        if dialog["task"] == "restaurant":
            ent_idx_res = dialog["gold_entity"]
        elif dialog["task"] == "attraction":
            ent_idx_att = dialog["gold_entity"]
        elif dialog["task"] == "hotel":
            ent_idx_hotel = dialog["gold_entity"]
        ent_index = list(set(ent_idx_res + ent_idx_att + ent_idx_hotel))
        dialog["ent_index"] = ent_index
        dialog["ent_idx_res"] = list(set(ent_idx_res))
        dialog["ent_idx_att"] = list(set(ent_idx_att))
        dialog["ent_idx_hotel"] = list(set(ent_idx_hotel))
        test_data.append(dialog)

    with open(entity_fp, 'r') as fr:
        global_entity = json.load(fr)
        global_entity_list = []
        for key in global_entity.keys():
                global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]
    global_entity_list = list(set(global_entity_list))

    F1_pred, F1_res_pred, F1_att_pred, F1_hotel_pred = 0, 0, 0, 0
    F1_count, F1_res_count, F1_att_count, F1_hotel_count = 0, 0, 0, 0
    TP_all, FP_all, FN_all = 0, 0, 0
    TP_res, FP_res, FN_res = 0, 0, 0
    TP_att, FP_att, FN_att = 0, 0, 0
    TP_hotel, FP_hotel, FN_hotel = 0, 0, 0

    for dialog in test_data:
        #pred_tokens = dialog["result"].replace('_', ' ').split()
        pred_tokens = dialog["result"].split()
        kb_arrys = dialog["kb"]

        gold_ents = dialog["ent_index"]
        tp, fp, fn, f1, count = compute_prf(gold_ents, pred_tokens, global_entity_list, kb_arrys)
        TP_all += tp
        FP_all += fp
        FN_all += fn
        F1_pred += f1
        F1_count += count

        gold_res_ents = dialog["ent_idx_res"]
        tp, fp, fn, f1, count = compute_prf(gold_res_ents, pred_tokens, global_entity_list, kb_arrys)
        TP_res += tp
        FP_res += fp
        FN_res += fn
        F1_res_pred += f1
        F1_res_count += count    

        gold_att_ents = dialog["ent_idx_att"]
        tp, fp, fn, f1, count = compute_prf(gold_att_ents, pred_tokens, global_entity_list, kb_arrys)
        TP_att += tp
        FP_att += fp
        FN_att += fn
        F1_att_pred += f1
        F1_att_count += count      

        gold_hotel_ents = dialog["ent_idx_hotel"]
        tp, fp, fn, f1, count = compute_prf(gold_hotel_ents, pred_tokens, global_entity_list, kb_arrys)
        TP_hotel += tp
        FP_hotel += fp
        FN_hotel += fn
        F1_hotel_pred += f1
        F1_hotel_count += count   
    
    if average == "micro":
        F1_score = compute_f1(TP_all, FP_all, FN_all)
        F1_res_score = compute_f1(TP_res, FP_res, FN_res)
        F1_att_score = compute_f1(TP_att, FP_att, FN_att)
        F1_hotel_score = compute_f1(TP_hotel, FP_hotel, FN_hotel)
    else:
        F1_score = F1_pred / float(F1_count)
        F1_res_score = F1_res_pred / float(F1_res_count)
        F1_att_score = F1_att_pred / float(F1_att_count)
        F1_hotel_score = F1_hotel_pred / float(F1_hotel_count)
        
    return F1_score, F1_res_score, F1_att_score, F1_hotel_score


def eval_entity_f1_camrest(data, entity_fp, average="micro"):
    test_data = []
    for dialog in data:
        #if len(dialog["gold_entity"]) > 0:
        #    dialog["gold_entity"] = ' '.join(dialog["gold_entity"]).replace('_', ' ').split()
        test_data.append(dialog)

    with open(entity_fp, 'r') as fr:
        global_entity = json.load(fr)
        global_entity_list = []
        for key in global_entity.keys():
            global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]
    global_entity_list = list(set(global_entity_list))

    F1_pred, F1_count = 0, 0
    TP_all, FP_all, FN_all = 0, 0, 0
    for dialog in test_data:
        #pred_tokens = dialog["result"].replace('_', ' ').split()
        pred_tokens = dialog["result"].split()
        kb_arrys = dialog["kb"]
        gold_ents = dialog["gold_entity"]
        tp, fp, fn, f1, count = compute_prf(gold_ents, pred_tokens, global_entity_list, kb_arrys)
        F1_pred += f1
        TP_all += tp
        FP_all += fp
        FN_all += fn
        F1_count += count
    if average == "micro":
        F1_score = compute_f1(TP_all, FP_all, FN_all)
    else:
        F1_score = F1_pred / float(F1_count)
       
    return F1_score


def get_result(task, dialog):
    # calculate bleu
    bleu = eval_bleu(dialog)

    # calculate acc
    resp_acc, dialog_acc = eval_dialog_accuracy(dialog)

    # calculate entity F1
    if task == 'kvr':
        entity_file = "data/KVR/kvret_entities.json"
        f1_score, f1_sch, f1_wet, f1_nav = eval_entity_f1_kvr(dialog, entity_file, average="micro")
        output_str = "BLEU SCORE: %.3f\n" % bleu
        output_str += "Per resp. ACC: %.2f%%\n" %(resp_acc * 100)
        output_str += "Per dialog ACC: %.2f%%\n" % (dialog_acc * 100)
        output_str += "F1 SCORE: %.2f%%\n" % (f1_score * 100)
        output_str += "Sch. F1: %.2f%%\n" % (f1_sch * 100)
        output_str += "Wet. F1: %.2f%%\n" % (f1_wet * 100)
        output_str += "Nav. F1: %.2f%%" % (f1_nav * 100)
        print(output_str)
    elif task == 'woz':
        entity_file = "data/MULTIWOZ/global_entities.json"
        f1_score, f1_res, f1_att, f1_hotel = eval_entity_f1_multiwoz(dialog, entity_file, average="micro")
        output_str = "BLEU SCORE: %.3f\n" % bleu
        output_str += "Per resp. ACC: %.2f%%\n" %(resp_acc * 100)
        output_str += "Per dialog ACC: %.2f%%\n" % (dialog_acc * 100)
        output_str += "F1 SCORE: %.2f%%\n" % (f1_score * 100)
        output_str += "Res. F1: %.2f%%\n" % (f1_res * 100)
        output_str += "Att. F1: %.2f%%\n" % (f1_att * 100)
        output_str += "Hot. F1: %.2f%%" % (f1_hotel * 100)
        print(output_str)
    elif task == 'cam':
        #entity_file = "data/CamRest/global_entities.json"
        entity_file = "data/CamRest_2/global_entities.json.oov"
        f1_score = eval_entity_f1_camrest(dialog, entity_file, average="micro")
        output_str = "BLEU SCORE: %.3f\n" % bleu
        output_str += "Per resp. ACC: %.2f%%\n" % (resp_acc * 100)
        output_str += "Per dialog ACC: %.2f%%\n" % (dialog_acc * 100)
        output_str += "F1 SCORE: %.2f%%" % (f1_score * 100)
        print(output_str)
    else:
        output_str = "No dataset available!"
        print(output_str)

    return bleu, f1_score

