"""Hyper-parameter search by Bayesian optimization
Usage:
  hyperparam_search.py --train_path=<filename> [--continue_log=<filename>]
  hyperparam_search.py (-h | --help)
  hyperparam_search.py --version
Options:
  -h --help                       Show this screen.
  --version                       Show version.
  --train_path=<filename>         Name of file to train (processed by sentencepeice)
  [--continue_log=<filename>]     Name of the json log file that we want the Bayesian optimization continues
"""


import os
import shutil
import re
import pickle
import sentencepiece as spm
import numpy as np
from datetime import datetime
from docopt import docopt
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import CountVectorizer

from mkprojections import create_projections
from hash import read_vocab, read_projections, projection, wta, hash_input
from log_classify import prepare_data, train_model

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs


def generate_projs(KC_size, proj_size):
    vocab, reverse_vocab, logprobs = read_vocab()

    PN_size = len(vocab)
    d = "models/kc"+str(KC_size)+"-p"+str(proj_size)
    if not os.path.isdir(d):
        os.mkdir(d)
    trial = len(os.listdir(d))
    model_file = create_projections(PN_size, KC_size, proj_size,d, trial)
    return model_file


def hash(model_file, in_file_path, output_dir, top_tokens, percent_hash):
    sp = spm.SentencePieceProcessor()
    sp.load('../spmcc.model')

    d = model_file
    vocab, reverse_vocab, logprobs = read_vocab()
    projection_functions, pn_to_kc = read_projections(d)
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=False, token_pattern='[^ ]+')

    # Setting up the fly
    PN_size = len(vocab)
    KC_size = len(projection_functions)
    proj_size = len(projection_functions[0])
    print("SIZES PN LAYER:",PN_size,"KC LAYER:",KC_size)
    print("SIZE OF PROJECTIONS:",proj_size)
    print("SIZE OF FINAL HASH:",percent_hash,"%")

    projection_layer = np.zeros(PN_size)
    kenyon_layer = np.zeros(KC_size)

    #Reading through documents
    n_doc = 0
    doc = ""

    M_data = []
    M_col = []
    M_row = []
    IDs = []
    classes = {}
    keywords = {}

    in_file = in_file_path.split('/')[-1]
    trial = d.split('.')[0].split('_')[1]
    params = '.kc'+str(KC_size) + '.size'+str(proj_size) + '.trial'+str(trial) + ".top"+str(top_tokens)+".wta"+str(percent_hash)

    hs_file = os.path.join(output_dir ,in_file.replace('.sp',params+'.hs')).replace('.projs/', '')
    ID_file = os.path.join(output_dir ,in_file.replace('.sp',params+'.ids')).replace('.projs/', '')
    class_file = os.path.join(output_dir ,in_file.replace('.sp',params+'.cls')).replace('.projs/', '')
    keyword_file = os.path.join(output_dir ,in_file.replace('.sp',params+'.kwords')).replace('.projs/', '')

    with open(in_file_path,'r') as f:
        for l in f:
            l = l.rstrip('\n')
            if l[:4] == "<doc":
                m = re.search(".*id=([^ ]*) ",l)
                ID=m.group(1)
                m = re.search(".*class=([^ ]*)>",l)
                cl=m.group(1)
                IDs.append(ID+'_'+cl)
                classes[IDs[-1]] = m.group(1)
            #                 print("Processing",IDs[-1])
            elif l[:5] == "</doc":
                #                 print(doc)
                ll = sp.encode_as_pieces(doc)
                X = vectorizer.fit_transform([doc])
                X = X.toarray()[0]
                vec = logprobs * X
                vec = wta(vec,top_tokens)
                hs = hash_input(vec,reverse_vocab,percent_hash, KC_size, pn_to_kc, projection_functions)
                hs = coo_matrix(hs)
                #                 keywords[IDs[-1]] = [reverse_vocab[w] for w in return_keywords(vec)]
                #                 print(keywords[IDs[-1]])
                for i in range(len(hs.data)):
                    M_row.append(n_doc)
                    M_col.append(hs.col[i])
                    M_data.append(hs.data[i])
                doc = ""
                n_doc+=1
                #time.sleep(0.002)    #Sleep a little to consume less CPU
            else:
                doc+=l+' '
    M = coo_matrix((M_data, (M_row, M_col)), shape=(n_doc, KC_size))

    with open(hs_file,"wb") as hsf:
        pickle.dump(M,hsf)
    with open(ID_file,"wb") as IDf:
        pickle.dump(IDs,IDf)
    with open(keyword_file,"wb") as kf:
        pickle.dump(keywords,kf)
    with open(class_file,"wb") as cf:
        pickle.dump(classes,cf)

    return hs_file


def classify(tr_file,C,num_iter,config):
    tr_file = tr_file.replace('./', '')
    dataset_name = tr_file.split('/')[-1].split('-')[0]
    now = datetime.now()

    m_train,classes_train,m_val,classes_val,ids_train,ids_val = prepare_data(tr_file)

    score = train_model(m_train,classes_train,m_val,classes_val,C,num_iter)
    with open('./log/results.tsv', 'a') as f:
        tmp = tr_file.split('.')
        kc = tmp[1][2:]
        size = tmp[2][4:]
        top = tmp[4][3:]
        wta = tmp[5][3:]
        l = '\t'.join([now.strftime("%Y-%m-%d %H:%M:%S"), dataset_name,
                       kc, size, top, wta, str(C), str(num_iter), str(score)])
        f.writelines(l + '\n')

    return score


def fruitfly_pipeline(train_path, topword, KC_size, proj_size, percent_hash, C, num_iter):
    model_file = generate_projs(KC_size, proj_size)
    config = "kc"+str(KC_size)+"-p"+str(proj_size)+"-h"+str(percent_hash)+"-C"+str(C)+"-i"+str(num_iter)
    os.mkdir("tmp/"+config)
    print('hashing files')
    hash_file = hash(model_file=model_file, in_file_path=train_path,
                     output_dir="tmp/"+config, top_tokens=topword, percent_hash=percent_hash)
    val_path = train_path.replace('train', 'val')
    hash(model_file=model_file, in_file_path=val_path,
         output_dir="tmp/"+config, top_tokens=topword, percent_hash=percent_hash)
    print('training and evaluating')
    val_score = classify(tr_file=hash_file, C=C, num_iter=num_iter, config=config)
    return val_score


def optimize_fruitfly(train_path, continue_log=''):
    def classify_val(topword, KC_size, proj_size, percent_hash,C):
        topword = round(topword)
        KC_size = round(KC_size)
        proj_size = round(proj_size)
        percent_hash = round(percent_hash)
        C = round(C)
        num_iter = 1000
        return fruitfly_pipeline(train_path, topword, KC_size,
                                 proj_size, percent_hash, C, num_iter)

    optimizer = BayesianOptimization(
        f=classify_val,
        pbounds={"topword": (10, 100), "KC_size": (3000, 9000),
            "proj_size": (3, 10), "percent_hash": (2, 20), "C": (1,100)},
        #random_state=1234,
        verbose=2
    )

    logger = JSONLogger(path="./log/logs.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    if continue_log:
        load_logs(optimizer, logs=[continue_log])
        print("Optimizer is now aware of {} points.".format(len(optimizer.space)))

    optimizer.maximize(n_iter=200)

    print("Final result:", optimizer.max)


if __name__ == '__main__':
    args = docopt(__doc__, version='Hyper-parameter search by Bayesian optimization, ver 0.1')
    train_path = args["--train_path"]
    continue_log = args["--continue_log"]
    if continue_log:
        optimize_fruitfly(train_path, continue_log)
    else:
        optimize_fruitfly(train_path)

