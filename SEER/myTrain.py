import math
from tqdm import tqdm

from utils.config import *
from models.SEER import *

early_stop = args['earlyStop']
if args['dataset']=='kvr':
    from utils.utils_Ent_kvr import *
elif args['dataset']=='cam':
    from utils.utils_Ent_cam import *
else:
    print("[ERROR] You need to provide the --dataset information")

# Fix random seed
if int(args['fix'])==1:
    def setup_seed(seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed) #cpu
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)  #并行gpu
            torch.backends.cudnn.deterministic = True  
            torch.backends.cudnn.benchmark = True   
    setup_seed(21)

# Configure models and load data
avg_best, cnt, acc = 0.0, 0, 0.0
train, dev, test, vocab, vocab_attribute, max_resp_len = prepare_data_seq(batch_size=int(args['batch']), OOVTest=args['OOVTest'])

model = EER(args['dataset'],
              vocab,
              vocab_attribute,
              max_resp_len,
              int(args['hidden']),
              int(args['hidden']),
              lr=float(args['learn']), 
              dropout=float(args['drop']),
              path=args['path'],
              B=int(args['relation_size_reduce']),
              share_embedding=args["shareDec"])

for epoch in range(30):
    print("Epoch:{}".format(epoch))  
    # Run the train function
    schedule_sampling = args['schedule_sampling_ratio'] / (args['schedule_sampling_ratio'] + math.exp(epoch / args['schedule_sampling_ratio']) - 1)
    print("schedule_sampling_ratio: ", schedule_sampling)

    pbar = tqdm(enumerate(train),total=len(train))
    for i, data in pbar:
        model.train_batch(data, int(args['clip']), reset=(i==0), ss=schedule_sampling)
        pbar.set_description(model.print_loss())

    if((epoch+1) % int(args['evalp']) == 0):    
        acc = model.evaluate(test, avg_best, early_stop)
        model.scheduler.step(acc)

        if(acc >= avg_best):
            avg_best = acc
            cnt = 0
        else:
            cnt += 1

        #if(cnt == 10 or (acc==1.0 and early_stop==None)): 
        #    print("Ran out of patient, early stop...")  
        #    break 


path = model.best
print("Testing path:", path)

model = EER(args['dataset'],
              vocab,
              vocab_attribute,
              max_resp_len,
              int(args['hidden']),
              int(args['hidden']),
              lr=float(args['learn']), 
              dropout=float(args['drop']),
              path=path,
              B=int(args['relation_size_reduce']),
              share_embedding=args["shareDec"])
model.evaluate(test, 1e7)

