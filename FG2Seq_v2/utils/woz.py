import json
import torch
import torch.utils.data as data
import torch.nn as nn
from utils.config import *
import ast

from utils.utils_general import *

def read_langs(file_name, slot_value_dict, max_line = None):
    print(("Reading lines from {}".format(file_name)))
    
    value_slot_dict = {}
    for slot in slot_value_dict.keys():
        for value in slot_value_dict[slot]:
            value_slot_dict[value] = slot

    max_resp_len = 0
    dummy = PAD_token
    data = []
    with open(file_name) as fin:
        cnt_lin, sample_counter = 1, 1
        ct_arr, kb_arr = [], []
        for line in fin:
            line = line.strip()
            if line:
                if line.startswith('#'):
                    task_type = line.replace('#', '')
                elif line.startswith('0'):
                    triple = line.split(' ')[1:]
                    #if triple[0] == triple[-1]: continue
                    if len(triple) != 3: continue
                    kb_arr.append(triple)
                else:
                    nid, line = line.split(' ', 1)
                    u, r, gold_ent = line.split('\t')
                    ct_arr.append(u.split(' '))
                
                    # Get gold entity for each domain
                    gold_ent = ast.literal_eval(gold_ent)

                    # Get entity list
                    value_list, slot_list = generate_entity_from_kb(kb_arr, value_slot_dict)
                    value_list, slot_list = generate_entity_from_ct(ct_arr, value_slot_dict, value_list, slot_list)

                    value_list = ["<global_kb>"] + value_list
                    slot_list = ["dummy"] + slot_list

                    response = r.split(' ')
                    sketch_response, ptr_index = generate_template(response, value_list, slot_list)
                    
                    #filter
                    kb_arr = [kb for kb in kb_arr if kb[0] != kb[-1]]

                    data_detail = {
                        'ct_arr':list(ct_arr),
                        'kb_arr':list(kb_arr),
                        'entity_list':value_list,
                        'response':response,
                        'sketch_response':sketch_response + ['EOS'],
                        'ptr_index':ptr_index + [dummy],
                        'ent_index':gold_ent,
                        'domain': task_type,
                        'id':int(sample_counter),
                        'ID':int(cnt_lin)}

                    data.append(data_detail)
                    
                    ct_arr.append(r.split(' '))

                    if max_resp_len < len(r.split()):
                        max_resp_len = len(r.split())
                    sample_counter += 1
            else:
                cnt_lin += 1
                ct_arr, kb_arr = [], []
                if(max_line and cnt_lin >= max_line):
                    break


    return data, max_resp_len


def generate_template(response, entity_list, entity_type_list, dummy_index=PAD_token):
    """
    Based on the system response and the provided entity table, the output is the sketch response. 
    """
    sketch_response = []
    ptr_index = []
    for word in response:
        if word not in entity_list:
            sketch_response.append(word)
            ptr_index.append(dummy_index)
        else:
            index = entity_list.index(word)
            sketch_response.append('@'+entity_type_list[index])
            ptr_index.append(index)

    return sketch_response, ptr_index


def generate_entity_from_kb(kb_arr, global_entity, entity_list=None, entity_type_list=None):
    if not entity_list: entity_list = []
    if not entity_type_list: entity_type_list = []

    for key_value, slot, value in kb_arr:
        if key_value not in entity_list:
            entity_list.append(key_value)
            entity_type_list.append("name")

        if value not in entity_list:
            entity_list.append(value)
            entity_type_list.append(slot)

    return entity_list, entity_type_list


def generate_entity_from_ct(ct_arr, global_entity, entity_list=None, entity_type_list=None):
    if not entity_list: entity_list = []
    if not entity_type_list: entity_type_list = []

    for sent in ct_arr:
        for word in sent:
            if word in entity_list:
                continue

            if word in global_entity:
                entity_list.append(word)
                entity_type_list.append(global_entity[word])


    return entity_list, entity_type_list


def prepare_data_seq(batch_size=100):
    file_train = 'data/MULTIWOZ/train.txt'
    file_dev = 'data/MULTIWOZ/dev.txt'
    file_test = 'data/MULTIWOZ/test.txt'
    entity_path = 'data/MULTIWOZ/global_entities.json'

    global_entities = json.load(open(entity_path, "r"))
    for k, v in global_entities.items():
        f = lambda x: x.lower().replace(' ', '_')
        global_entities[k] = list(map(f, v))

    pair_train, train_max_len = read_langs(file_train, global_entities, max_line=None)
    pair_dev, dev_max_len = read_langs(file_dev, global_entities, max_line=None)
    pair_test, test_max_len = read_langs(file_test, global_entities, max_line=None)
    max_resp_len = max(train_max_len, dev_max_len, test_max_len) + 1
    
    # dictionary for vocabulary and relation
    vocabulary = Lang()
    relation = Lang(init=False)
    for data in pair_train:
        vocabulary.index_words(data["ct_arr"], trg=False)
        vocabulary.index_words(data['entity_list'], trg=True)
        vocabulary.index_words(data["response"], trg=True)
        vocabulary.index_words(data["sketch_response"], trg=True)

        for _, slot, _ in data["kb_arr"]:
            relation.index_word(slot)

    lang = (vocabulary, relation)

    train = get_seq(pair_train, lang, batch_size, True)
    dev   = get_seq(pair_dev, lang, batch_size, False)
    test  = get_seq(pair_test, lang, batch_size, False)
    
    print("Read %s sentence pairs train" % len(pair_train))
    print("Read %s sentence pairs dev" % len(pair_dev))
    print("Read %s sentence pairs test" % len(pair_test))
    print("Vocabulary size: %s " % vocabulary.n_words)
    print("Relation size: %s " % relation.n_words)
    print("Max. length of system response: %s " % max_resp_len)
    print("USE_CUDA={}".format(USE_CUDA))
    
    return train, dev, test, lang, max_resp_len

