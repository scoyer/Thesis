import json
import torch
import torch.nn as nn
import ast

from utils.config import *
from utils.utils_general import *

table_column = {
    "navigate": ["distance","traffic_info","poi_type","address","poi"],
    "schedule": ['time','date','party','room','agenda','event'],
    "weather": ["today",
                "monday_weather", "monday_low", "monday_high",
                "tuesday_weather", "tuesday_low", "tuesday_high",
                "wednesday_weather", "wednesday_low", "wednesday_high",
                "thursday_weather", "thursday_low", "thursday_high",
                "friday_weather", "friday_low", "friday_high",
                "saturday_weather", "saturday_low", "saturday_high",
                "sunday_weather", "sunday_low", "sunday_high",
                "location"]
}

table_slot = {
    "navigate": ["@distance","@traffic_info","@poi_type","@address","@poi"],
    "schedule": ['@time','@date','@party','@room','@agenda','@event'],
    "weather": ["@date",
                "@weather_attribute", "@temperature", "@temperature",
                "@weather_attribute", "@temperature", "@temperature",
                "@weather_attribute", "@temperature", "@temperature",
                "@weather_attribute", "@temperature", "@temperature",
                "@weather_attribute", "@temperature", "@temperature",
                "@weather_attribute", "@temperature", "@temperature",
                "@weather_attribute", "@temperature", "@temperature",
                "@location"]
}

def read_langs(file_name, max_line = None, global_file='data/KVR/kvret_entities.json', build_vocab=False, is_OT=False):
    print(("Reading lines from {}".format(file_name)), end="")
    
    with open(global_file, "r") as f:
        global_entity = json.load(f)
        global_entity_list = {}
        for key in global_entity.keys():
            if key != 'poi':
                if key not in global_entity_list:
                    global_entity_list[key] = []
                global_entity_list[key] += [item.lower().replace(' ', '_') for item in global_entity[key]]
            else:
                for item in global_entity['poi']:
                    for k in item.keys():
                        if k == "type":
                            continue
                        if k not in global_entity_list:
                            global_entity_list[k] = []
                        global_entity_list[k] += [item[k].lower().replace(' ', '_')]
    
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
                if '#' in line:
                    dialog_ID += 1
                    line = line.replace("#","")
                    task_type = line
                    continue

                nid, line = line.split(' ', 1)
                if int(nid) > 0:
                    u, r, gold_ent = line.split('\t')
                    context_arr.extend(u.split(' '))
                    response = r.split(' ')
                
                    # Get gold entity for each domain
                    gold_ent = ast.literal_eval(gold_ent)
                    ent_idx_cal, ent_idx_nav, ent_idx_wet = [], [], []
                    if task_type == "weather": ent_idx_wet = gold_ent
                    elif task_type == "schedule": ent_idx_cal = gold_ent
                    elif task_type == "navigate": ent_idx_nav = gold_ent
                    ent_index = list(set(ent_idx_cal + ent_idx_nav + ent_idx_wet))

                    # Get entity set from context and kb
                    entity_set = ["PAD"]  # represent empty token
                    entity_set_type = ["<null>"]

                    entity_set, entity_set_type = generate_entity_from_kb(kb_arr, task_type, \
                                                                entity_set, entity_set_type)
                    entity_set, entity_set_type = generate_entity_from_context(context_arr, global_entity_list, \
                                                                entity_set, entity_set_type)

                    entity_set.append("<null>")  # special token
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

                    # Add a <null> in context entity
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
                            if entity not in entity_set:
                                continue
                            index = entity_set.index(entity)
                            row_index = len(kb_entity_id)
                            col_attribute = table_column[task_type][col_id]
                            kb_entity_id[-1].append((index, row_index, col_attribute))

                    # Add a <null> in kb entity
                    kb_entity_id.append([(len(entity_set)-1, len(kb_entity_id)+1, "<null>")])

                    if len(kb_entity_id) < 2:
                        kb_entity_id.append([(0, len(kb_entity_id)+1, "<null>")])

                    # Generate sketch response whose entity value is replaced by slot
                    sketch_response = generate_template(response, gold_ent, global_entity_list)

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
                        'ent_idx_cal':list(set(ent_idx_cal)),
                        'ent_idx_nav':list(set(ent_idx_nav)),
                        'ent_idx_wet':list(set(ent_idx_wet)),
                        'dialog_id':int(dialog_id),
                        'dialog_ID':int(dialog_ID),
                        'domain':task_type,
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
                context_arr, kb_arr = [], []
                if(max_line and cnt_lin >= max_line):
                    break
    print("  num of dialogs:", dialog_ID)

    if build_vocab:
        vocab_attribute.index_words(list(table_column.values()))
        vocab_attribute.index_word("<null>")

        if is_OT:
            filter_word_list = global_entity_list['poi']
            filter_word_list += global_entity_list['address']
            filter_word_list += global_entity_list['event']
            filter_word_list += global_entity_list['party']
            filter_word_list += global_entity_list['location']
            vocab.delete_words(filter_word_list)

        return data, max_resp_len, vocab, vocab_attribute

    return data, max_resp_len


def generate_entity_from_kb(kb_arr, task_type, entity_set, entity_set_type):
    for kb in kb_arr:
        for entity_id, entity in enumerate(kb):
            if entity == "<empty>":
                continue
            if entity not in entity_set:
                entity_set.append(entity)
                entity_set_type.append(table_slot[task_type][entity_id])
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


def generate_template(response, sent_ent, global_entity):
    """
    Based on the system response and the provided entity table, the output is the sketch response. 
    """
    sketch_response = [] 
    if sent_ent == []:
        sketch_response = list(response)
    else:
        for word in response:
            if word not in sent_ent:
                sketch_response.append(word)
            else:
                ent_type = None
                for k, v in global_entity.items():
                    if word in v:
                        ent_type = k
                        break

                assert ent_type != None, word
                sketch_response.append('@'+ent_type)

    return sketch_response


def prepare_data_seq(batch_size=100, OOVTest=False):
    file_train = 'data/KVR/train.txt'
    file_dev = 'data/KVR/dev.txt'
    file_test = 'data/KVR/test.txt'

    pair_train, train_max_len, vocab, vocab_attribute = read_langs(file_train, max_line=None, build_vocab=True, is_OT=OOVTest)
    pair_dev, dev_max_len = read_langs(file_dev, max_line=None)
    pair_test, test_max_len = read_langs(file_test, max_line=None)
    max_resp_len = max(train_max_len, dev_max_len, test_max_len) + 1
    
    map_table = {}
    if OOVTest:
        with open("data/KVR/kvret_entities.json","r") as f:
            global_entity = json.load(f)

        global_entity_list = {}
        for key in global_entity.keys():
            if key != 'poi':
                if key not in global_entity_list:
                    global_entity_list[key] = []
                global_entity_list[key] += [item.lower().replace(' ', '_') for item in global_entity[key]]
            else:
                for item in global_entity['poi']:
                    for k in item.keys():
                        if k == "type":
                            continue
                        if k not in global_entity_list:
                            global_entity_list[k] = []
                        global_entity_list[k] += [item[k].lower().replace(' ', '_')]
    
        for k,v in global_entity_list.items():
            if k in ["poi", "address", "event",  "party", "location"]:
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
