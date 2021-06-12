from utils.config import *
from models.FG2Seq import *

directory = args['path'].split("/")
DS = directory[1].split('DS')[1].split('HDD')[0]
HDD = directory[1].split('HDD')[1].split('BSZ')[0]
BSZ =  int(directory[1].split('BSZ')[1].split('DR')[0])
B = directory[1].split('RS')[1].split('BLEU')[0]

if DS=='kvr': 
    from utils.kvr import *
elif DS=='cam':
    from utils.cam import *
    #file_test = 'data/CamRest/test.txt'
    #entity_path = 'data/CamRest/global_entities.json'
    file_test = 'data/CamRest_2/test.txt.oov'
    entity_path = 'data/CamRest_2/global_entities.json.oov'
elif DS=='woz':
    from utils.woz import *
else: 
    print("You need to provide the --dataset information")


train, dev, test, lang, max_resp_len = prepare_data_seq(batch_size=BSZ)

global_entities = json.load(open(entity_path, "r"))
for k, v in global_entities.items():
    f = lambda x: x.lower().replace(' ', '_')
    global_entities[k] = list(map(f, v))

pair_test, _ = read_langs(file_test, global_entities, max_line=None)
test  = get_seq(pair_test, lang, BSZ, False)


model = FG2Seq(
	int(HDD), 
	lang, 
	max_resp_len, 
	args['path'], 
	DS, 
	lr=0.0, 
	dropout=0.0,
        relation_size=lang[1].n_words,
        B=int(B),
        share_embedding=args["shareDec"])

acc_test = model.evaluate(test, 1e7) 
