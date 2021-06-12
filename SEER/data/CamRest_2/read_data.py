import ast
import os
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
f = open('camrest_entities.json','r',encoding='utf-8')
slot_value_dict = json.load(f)


replace_dict = {}
#replace_keys = ["name", "address", "phone", "location", "postcode"]
replace_keys = ["name", "phone", "postcode"]
for key in replace_keys:
    for ent in slot_value_dict[key]:
        replace_dict[ent] = key + "_" + ent


for key in replace_keys:
    slot_value_dict[key] = [key + "_" + e for e in slot_value_dict[key]]

nf = open('camrest_entities.json.oov',"w")
json.dump(slot_value_dict, nf, indent=4)

write_file("train.txt", "train.txt.oov", replace_dict)
write_file("dev.txt", "dev.txt.oov", replace_dict)
write_file("test.txt", "test.txt.oov", replace_dict)

