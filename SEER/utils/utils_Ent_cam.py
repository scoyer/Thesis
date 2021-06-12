import json
import torch
import torch.nn as nn
import ast

from utils.config import *
from utils.utils_general import *

table_column = ["address", "area", "food", "location", "phone", "postcode", "pricerange", "name"]
table_slot = ["@address", "@area", "@food", "@location", "@phone", "@postcode", "@pricerange", "@name"]

def read_langs(file_name, max_line = None, global_file='data/CamRest/camrest_entities.json', build_vocab=False, is_OT=False):
    print(("Reading lines from {}".format(file_name)),end="")
    
    with open(global_file, "r") as f:
        global_entity_list = json.load(f)

    if build_vocab:
        vocab = Vocab()
        vocab_attribute = Vocab(is_for_attribute=True)

    max_resp_len = 0
    data, context_arr, kb_arr = [], [], []
    dialog_id, dialog_ID = 0, 0
    with open(file_name) as fin:
        for line in fin:
            line = line.strip()
            if line:
                nid, line = line.split(' ', 1)
                if int(nid) > 0:
                    u, r, gold_ent = line.split('\t')
                    context_arr.extend(u.split(' '))
                    response = r.split(' ')
                
                    # Get gold entity for each domain
                    gold_ent = ast.literal_eval(gold_ent)
                    ent_index = list(set(gold_ent))

                    # Get entity set from context and kb
                    entity_set = ["PAD"]  # represent empty token
                    entity_set_type = ["<null>"]

                    entity_set, entity_set_type = generate_entity_from_kb(kb_arr, entity_set, entity_set_type)
                    entity_set, entity_set_type = generate_entity_from_context(context_arr, global_entity_list, \
                                                                            entity_set, entity_set_type)


                    entity_set.append("<null>")  # represent special token
                    entity_set_type.append("<null>")

                    # Get local pointer position for each word in system response
                    response_entity_id = []
                    for key in response:
                        if key in entity_set:
                            index = entity_set.index(key)
                        else:
                            index = len(entity_set) - 1 #it represents <null> token
                        response_entity_id.append(index)

                    # Get context entity position and context entity index of entity_set
                    context_entity_pos, context_entity_id = [], []
                    for word_id, word in enumerate(context_arr):
                        if word in entity_set:
                            context_entity_pos.append(word_id)
                            context_entity_id.append(entity_set.index(word))

                    # Add two <null> as context entity whose position is 0 and len - 1 respectively
                    context_entity_pos.append(len(context_arr) - 1)
                    context_entity_id.append(len(entity_set) - 1)

                    if len(context_entity_pos) < 2:
                        context_entity_pos.append(0)
                        context_entity_id.append(0)

                    # Get kb entity index of entity_set
                    kb_entity_id = []
                    for kb_row in kb_arr:
                        kb_entity_id.append([])
                        for col_id, entity in enumerate(kb_row):
                            index = entity_set.index(entity)
                            row_index = len(kb_entity_id)
                            col_attribute = table_column[col_id]
                            kb_entity_id[-1].append((index, row_index, col_attribute))

                    # Add a <null> in kb entity
                    kb_entity_id.append([(len(entity_set)-1, len(kb_entity_id)+1, "<null>")])

                    if len(kb_entity_id) < 2:
                        kb_entity_id.append([(0, len(kb_entity_id)+1, "<null>")])

                    # Generate sketch response whose entity value is replaced by slot
                    sketch_response = generate_template(response, ent_index, entity_set, entity_set_type)

                    # Generate indicator that represents if the entity apears in the dialog history
                    indicator = [1 if entity in context_arr else 0 for entity in entity_set]

                    data_detail = {
                        'context':list(context_arr),
                        'entity':entity_set,
                        'entity_type':entity_set_type,
                        'response':list(response),
                        'sketch_response':list(sketch_response),
                        'context_entity_pos': context_entity_pos,
                        'context_entity_id': context_entity_id,
                        'kb_entity_id': kb_entity_id,
                        'response_entity_id':response_entity_id+[len(entity_set) - 1],
                        'indicator':indicator,
                        'ent_index':ent_index,
                        'dialog_id':int(dialog_id),
                        'dialog_ID':int(dialog_ID),
                        'domain':"#",
                    }
                    data.append(data_detail)
                    
                    context_arr.extend(response)
                    dialog_id += 1
                    if max_resp_len < len(response):
                        max_resp_len = len(response)

                    if build_vocab:
                        vocab.index_words(context_arr)
                        vocab.index_words(entity_set)
                        vocab.index_words(response)
                        vocab.index_words(sketch_response)
                else:
                    kb = line.split('\t')
                    kb_arr.append(kb)
            else:
                dialog_ID += 1
                context_arr, kb_arr = [], []
                if(max_line and cnt_lin >= max_line):
                    break

    print(" num of dialogs:", dialog_ID)
    
    if build_vocab:
        vocab_attribute.index_words(table_column)
        vocab_attribute.index_word("<null>")

        if is_OT:
            filter_word_list = global_entity_list['name'] + global_entity_list["phone"] + global_entity_list["postcode"]
            vocab.delete_words(filter_word_list)

        return data, max_resp_len, vocab, vocab_attribute

    return data, max_resp_len


def generate_entity_from_kb(kb_arr, entity_set, entity_set_type):
    for kb in kb_arr:
        assert len(kb) == len(table_slot)
        for entity_id, entity in enumerate(kb):
            if entity == "<empty>":
                continue

            if entity not in entity_set:
                entity_set.append(entity)
                entity_set_type.append(table_slot[entity_id])

    return entity_set, entity_set_type


def generate_entity_from_context(context_arr, global_entity, entity_set, entity_set_type):
    for word in context_arr:
        if word in entity_set:
            continue

        for k, v in global_entity.items():
            if word in v:
                entity_set.append(word)
                entity_set_type.append('@'+k)
                break

    return entity_set, entity_set_type


def generate_template(response, sent_ent, entity_set, entity_set_type):
    """
    Based on the system response and the provided entity table, the output is the sketch response. 
    """
    sketch_response = [] 
    if sent_ent == []:
        sketch_response = list(response)
    else:
        for word in response:
            if word not in entity_set:
                sketch_response.append(word)
            else:
                index = entity_set.index(word)
                ent_type = entity_set_type[index]
                sketch_response.append(ent_type)
    return sketch_response


def prepare_data_seq(batch_size=100, OOVTest=False):
    file_train = 'data/CamRest/train.txt'
    file_dev = 'data/CamRest/dev.txt'
    file_test = 'data/CamRest/test.txt'

    pair_train, train_max_len, vocab, vocab_attribute = read_langs(file_train, max_line=None, build_vocab=True, is_OT=OOVTest)
    pair_dev, dev_max_len = read_langs(file_dev, max_line=None)
    pair_test, test_max_len = read_langs(file_test, max_line=None)
    max_resp_len = max(train_max_len, dev_max_len, test_max_len) + 1

    map_table = {}
    if OOVTest:
        with open("data/CamRest/camrest_entities.json","r") as f:
            global_entity_list = json.load(f)
        for k,v in global_entity_list.items():
            if k in ["name", "phone", "postcode"]:
                for _v in v:
                    map_table[_v] = 'token_'+k
                    vocab.index_word('token_'+k)
                    #map_table[_v] = '@'+k
                    
    
    train = get_seq(pair_train, vocab, vocab_attribute, batch_size, True, map_table)
    dev   = get_seq(pair_dev, vocab, vocab_attribute, batch_size, False, map_table)
    test  = get_seq(pair_test, vocab, vocab_attribute, batch_size, False, map_table)
    
    print("Read %s sentence pairs train" % len(pair_train))
    print("Read %s sentence pairs dev" % len(pair_dev))
    print("Read %s sentence pairs test" % len(pair_test))
    print("Vocab_size: %s " % vocab.n_words)
    print("Max. length of system response: %s " % max_resp_len)
    print("USE_CUDA={}".format(USE_CUDA))
    
    return train, dev, test, vocab, vocab_attribute, max_resp_len
