from utils.config import *
from models.SEER import *
import os

directory = args['path'].split("/")
OT = True if 'OT' in directory[1] else False
DS = directory[1].split('DS')[1].split('HDD')[0]
HDD = directory[1].split('HDD')[1].split('BSZ')[0]
BSZ =  int(directory[1].split('BSZ')[1].split('DR')[0])
B = directory[1].split('RS')[1].split('BLEU')[0]

if DS=='kvr': 
    from utils.utils_Ent_kvr import *
    file_test = 'data/KVR_2/test.txt.oov'
    global_file='data/KVR_2/kvret_entities.json.oov'
    map_table = {}
elif DS=='cam':
    from utils.utils_Ent_cam import *
    file_test = 'data/CamRest_2/test.txt.oov'
    global_file='data/CamRest_2/camrest_entities.json.oov'
    map_table = {}
    if OT:
        with open(global_file,"r") as f:
            global_entity_list = json.load(f)
        for k,v in global_entity_list.items():
            if k in ["name", "phone", "postcode"]:
                for _v in v:
                    map_table[_v] = 'token_'+k
                    #map_table[_v] = '@'+k

else: 
    print("You need to provide the --dataset information")

train, dev, test, vocab, vocab_attribute, max_resp_len = prepare_data_seq(batch_size=BSZ, OOVTest=OT)

#pair_test, test_max_len = read_langs(file_test, global_file=global_file)
#test  = get_seq(pair_test, vocab, vocab_attribute, BSZ, False, map_table)

model = EER(DS, vocab, vocab_attribute, max_resp_len, int(HDD), int(HDD), lr=0.0,  dropout=0.0, path=args['path'], B=int(B), share_embedding=False)

acc_test = model.evaluate(test, 1e7)
