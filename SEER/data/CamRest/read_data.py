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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--json', dest='json',
                        default='CamRest676_train.json',
                        help='process json file')
    args = parser.parse_args()

    with open(args.json) as f:
        dialogues = json.load(f)

    with open('camrest_entities.json') as f:
        entities_dict = json.load(f)
    global_entity_list = list(entities_dict.values())
    global_entity_list = sum(global_entity_list, [])

    column_names = ["address", "area", "food", "location", "phone", "postcode", "pricerange", "name"]

    for d in dialogues:
        #kb
        for kb in d['scenario']['kb']['items']:
            output_string = []
            for slot in column_names:
                value = str(kb[slot].lower())
                output_string.append(value)

            output_string = "\t".join(output_string)
            print("0 {}".format(output_string))

        #dialog
        if (len(d['dialogue'])%2 != 0):
            d['dialogue'].pop()

        j = 1
        for i in range(0, len(d['dialogue']), 2):
            user = " ".join(cleaner(tokenizer(str(d['dialogue'][i]['data']['utterance']).lower())))
            bot = " ".join(cleaner(tokenizer(str(d['dialogue'][i+1]['data']['utterance']).lower())))
            gold_entity = []
            for key in bot.split(' '):
                if key in global_entity_list:
                    gold_entity.append(key)
                
            gold_entity = list(set(gold_entity))
            if user!="" and bot!="":
                print(str(j)+" "+user+'\t'+bot+'\t'+str(gold_entity))
                j+=1
        print("")
