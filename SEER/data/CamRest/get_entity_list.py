import os
import json
#from nltk import wordpunct_tokenize as tokenizer
import argparse

def tokenizer(sentence):
    return sentence.strip().split()

def cleaner(token_array):
    new_token_array = []
    for idx, token in enumerate(token_array):
        temp = token
        if token==".." or token=="." or token=="...": continue
        if token==":),": continue
        if token=="n't": temp="not"
        new_token_array.append(temp)
    return new_token_array

column_names = ["address", "area", "food", "location", "phone", "postcode", "pricerange", "name"]

def get_global_list(filename):
    with open(filename) as f:
        dialogues = json.load(f)

    global_list = {}
    for d in dialogues:
        #kb
        entity_set = []
        for kb in d['scenario']['kb']['items']:
            for c in column_names:
                #filter the useless columns
                #if c not in column_names:
                #    continue

                value = str(kb[c].lower())
                if c not in global_list:
                    global_list[c] = []

                global_list[c].append(value)
                
    return global_list


    
if __name__ == "__main__":
    train = "CamRest676_train.json"
    dev = "CamRest676_dev.json"
    test = "CamRest676_test.json"

    train_list = get_global_list(train)
    dev_list = get_global_list(dev)
    test_list = get_global_list(test)

    data = {}
    for c in column_names:
        data[c] = list(set(train_list[c] + dev_list[c] + test_list[c]))

    with open("camrest_entities.json", "w") as f:
        json.dump(data, f, indent=4)
        
