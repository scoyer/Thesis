import re
import os
import ast
import collections

def write_file(file_name, new_file, replace_dict):
    def replace_seq(seq):
        new_seq = []
        for s in seq:
            if s in replace_dict:
                new_seq.append(replace_dict[s])
            else:
                new_seq.append(s)

        return new_seq

    nf = open(new_file, "w")
    with open(file_name) as fin:
        for raw_line in fin:
            line = raw_line.strip()
            if line:
                if '#' in line:
                    nf.write(raw_line)
                    continue

                nid, line = line.split(' ', 1)
                if int(nid) > 0:
                    u, r, gold_ent = line.split('\t')
                    u = u.split(' ')
                    r = r.split(' ')
                    u = replace_seq(u)
                    r = replace_seq(r)
                    u = " ".join(u)
                    r = " ".join(r)

                    gold_ent = ast.literal_eval(gold_ent)
                    gold_ent = replace_seq(gold_ent)
                    gold_ent = str(gold_ent)

                    nf.write("{} {}\t{}\t{}\n".format(nid, u, r, gold_ent))
                else:
                    kb = line.split('\t')
                    kb = replace_seq(kb)
                    kb = "\t".join(kb)
                    nf.write("0 " + kb + "\n")
            else:
                nf.write(raw_line)


import json
with open('kvret_entities.json') as f:
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

slot_value_dict = global_entity_list

replace_dict = {}
replace_keys = ["poi", "address", "event", "location", "party"]
for key in replace_keys:
    for ent in slot_value_dict[key]:
        replace_dict[ent] = key + "_" + ent


for key in global_entity.keys():
    if key != "poi":
        if key in replace_keys:
            global_entity[key] = [key + " " + e for e in global_entity[key]]
    else:
        for item in global_entity["poi"]:
            for k in item.keys():
                if k == "type":
                    continue
                if k in replace_keys:
                    item[k] = k + " " + item[k]


nf = open('kvret_entities.json.oov',"w")
json.dump(global_entity, nf, indent=4)

write_file("train.txt", "train.txt.oov", replace_dict)
write_file("dev.txt", "dev.txt.oov", replace_dict)
write_file("test.txt", "test.txt.oov", replace_dict)

