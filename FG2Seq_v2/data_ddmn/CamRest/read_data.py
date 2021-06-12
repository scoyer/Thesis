import os
import collections

def read_file(file_name, slot_value_dict=collections.defaultdict(set)):
    with open(file_name, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                if line.startswith("0"):
                    slot, value = line.split(" ")[2:]
                    slot_value_dict[slot].add(value)
            else:
                pass

    return slot_value_dict

slot_value_dict = read_file("train.txt")
slot_value_dict = read_file("dev.txt", slot_value_dict)
slot_value_dict = read_file("test.txt", slot_value_dict)


import json
f = open('camrest676-entities.json','r',encoding='utf-8')
slot_value_old = json.load(f)
for k in slot_value_old:
    v1 = slot_value_old[k]

    for v in v1:
        slot_value_dict[k].add(v)

for k in slot_value_dict:
    slot_value_dict[k] = list(slot_value_dict[k])

nf = open('global_entities.json',"w")
json.dump(slot_value_dict, nf, indent=4)
