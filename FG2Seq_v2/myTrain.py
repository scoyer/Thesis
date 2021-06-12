from tqdm import tqdm

from utils.config import *
from utils.utils_general import get_schdule_sampling_rate
from models.FG2Seq import *


early_stop = args['earlyStop']
if args['dataset']=='kvr':
    from utils.kvr import *
    early_stop = 'BLEU'
elif args['dataset']=='cam':
    from utils.cam import *
    early_stop = 'BLEU'
elif args['dataset']=='woz':
    from utils.woz import *
    early_stop = 'BLEU'
else:
    print("[ERROR] You need to provide the --dataset information")

# Configure models and load data
avg_best, cnt, acc = 0.0, 0, 0.0
train, dev, test, lang, max_resp_len = prepare_data_seq(batch_size=int(args['batch']))

model = FG2Seq(
    int(args['hidden']), 
    lang, 
    max_resp_len, 
    args['path'], 
    args['dataset'], 
    lr=float(args['learn']), 
    dropout=float(args['drop']),
    relation_size=lang[1].n_words,
    B=int(args['relation_size_reduce']),
    share_embedding=args["shareDec"])

#optimizer = torch.optim.Adam(model.parameters(), lr=float(args['learn']))
#scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, min_lr=0.0001, verbose=True)

for epoch in range(200):
    print("Epoch:{}".format(epoch))  
    # Run the train function
    ss_rate = get_schdule_sampling_rate(epoch, args['schedule_sampling_ratio'])
    print("schedule_sampling_ratio: ", ss_rate)

    pbar = tqdm(enumerate(train),total=len(train))
    for i, data in pbar:
        #model.train_batch(optimizer, data, int(args['clip']), reset=(i==0), ss=ss_rate)
        model.train_batch(data, int(args['clip']), reset=(i==0), ss=ss_rate)
        pbar.set_description(model.print_loss())
        # break
    if((epoch+1) % int(args['evalp']) == 0):    
        acc = model.evaluate(dev, avg_best, early_stop)
        model.scheduler.step(acc)
        #scheduler.step(acc)

        if(acc >= avg_best):
            avg_best = acc
            cnt = 0
        else:
            cnt += 1

        if(cnt == 10 or (acc==1.0 and early_stop==None)): 
            print("Ran out of patient, early stop...")  
            break 


path = model.best
print("Testing path:", path)
model = FG2Seq(
    int(args['hidden']), 
    lang, 
    max_resp_len, 
    path, 
    args['dataset'], 
    lr=float(args['learn']), 
    dropout=float(args['drop']),
    relation_size=lang[1].n_words,
    B=int(args['relation_size_reduce']),
    share_embedding=args["shareDec"])

model.evaluate(test, 1e7)
