from bert_serving.client import BertClient
from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer
from tqdm import tqdm
import json
import multiprocessing


MAX_SEQLEN = 16
MAX_TEXTLEN = 25
BERT_MODEL_PATH = '../../embedding/models/multi_cased_L-12_H-768_A-12'

# bert-serving-start -model_dir jiayunz/Structure_Hole/embedding/models/multi_cased_L-12_H-768_A-12 -max_seq_len 25 -max_batch_size 16 -num_worker 4 -cased_tokenization -cpu -pooling_strategy NONE

def start_bert_server():
    args = get_args_parser().parse_args(['-model_dir', BERT_MODEL_PATH,
                                         '-max_seq_len', str(MAX_TEXTLEN),
                                         '-max_batch_size', str(MAX_SEQLEN),
                                         #'-pooling_strategy', 'NONE',
                                         '-num_worker', str(multiprocessing.cpu_count()),
                                         '-port', '5555',
                                         '-port_out', '5556',
                                         '-cased_tokenization',
                                         '-cpu'])
    server = BertServer(args)
    server.start()

def get_sentence_embedding(rpath, wpath):
    bc = BertClient()
    with open(wpath, 'w') as wf:
        with open(rpath, 'r') as rf:
            lines = rf.readlines()
            for line in tqdm(lines, total=len(lines)):
                user = json.loads(line.strip())
                tips = [t['text'] for t in user['fsq']['tips']['tips content'][:MAX_SEQLEN]]
                emb_tips = bc.encode(tips)
                user['fsq']['tips']['tips embedding'] = emb_tips.tolist()
                wf.write(json.dumps(user) + '\n')
    BertServer.shutdown()

def get_word_embedding(rpath, wpath):
    args = get_args_parser().parse_args(['-model_dir', BERT_MODEL_PATH,
                                         '-max_seq_len', str(MAX_TEXTLEN),
                                         '-max_batch_size', str(MAX_SEQLEN),
                                         '-pooling_strategy', 'NONE',
                                         '-num_worker', '8',
                                         '-port', '5555',
                                         '-port_out', '5556',
                                         '-cased_tokenization',
                                         '-cpu'])
    server = BertServer(args)
    server.start()
    bc = BertClient()
    with open(wpath, 'w') as wf:
        with open(rpath, 'r') as rf:
            lines = rf.readlines()
            for line in tqdm(lines, total=len(lines)):
                user = json.loads(line.strip())
                tips = [t['text'] for t in user['fsq']['tips']['tips content'][:MAX_SEQLEN]]
                emb_tips = bc.encode(tips)
                user['fsq']['tips']['tips embedding'] = emb_tips.tolist()
                wf.write(json.dumps(user) + '\n')
    BertServer.shutdown(args)

if __name__ == '__main__':
    #get_sentence_embedding('/bdata/jiayunz/Foursquare/100w/train_1.json', '/bdata/jiayunz/Foursquare/100w/train_1_sentence_emb.json')
    #get_sentence_embedding('/bdata/jiayunz/Foursquare/100w/test_1.json', '/bdata/jiayunz/Foursquare/100w/test_1_sentence_emb.json')
    get_word_embedding('/bdata/jiayunz/Foursquare/100w/train_1_sentence_emb.json', '/bdata/jiayunz/Foursquare/100w/train_1_word_emb.json')
    get_word_embedding('/bdata/jiayunz/Foursquare/100w/test_1_sentence_emb.json', '/bdata/jiayunz/Foursquare/100w/test_1_word_emb.json')