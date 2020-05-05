import os, numpy as np
import scipy.sparse as sp
from gensim.corpora import Dictionary as gensim_dico

def parse_processed_amazon_dataset(FNames, max_words):
    datasets = {}
    dico = gensim_dico()

    # First pass on document to build dictionary
    for fname in FNames:
        f = open(fname)
        for l in f:
            tokens = l.split(sep=' ')
            label_string = tokens[-1]
            tokens_list=[]
            for tok in tokens[:-1]:
                ts, tfreq = tok.split(':')
                freq = int(tfreq)
                tokens_list += [ts]*freq

            _ = dico.doc2bow(tokens_list, allow_update=True)

        f.close()

    # Preprocessing_options
    dico.filter_extremes(no_below=2, keep_n=max_words)
    dico.compactify()

    for fname in FNames:
        X = []
        Y = []
        docid = -1
        f = open(fname)
        for l in f:
            tokens = l.split(sep=' ')
            label_string = tokens[-1]
            tokens_list = []
            for tok in tokens[:-1]:
                ts, tfreq = tok.split(':')
                freq = int(tfreq)
                tokens_list += [ts]*freq

            count_list = dico.doc2bow(tokens_list, allow_update=False)

            docid += 1

            X.append((docid, count_list))

            # Preprocess Label
            ls, lvalue = label_string.split(':')
            if ls == "#label#":
                if lvalue.rstrip() == 'positive':
                    lv = 1
                    Y.append(lv)
                elif lvalue.rstrip() == 'negative':
                    lv = 0
                    Y.append(lv)
                else:
                    raise Exception("Invalid Label Value")
            else:
                raise Exception('Invalid Format')

        datasets[fname] = (X, np.array(Y))
        f.close()
        del f

    return datasets, dico


def count_list_to_sparse_matrix(X_list, dico):
    ndocs = len(X_list)
    voc_size = len(dico.keys())


    X_spmatrix = sp.lil_matrix((ndocs, voc_size))
    for did, counts in X_list:
        for wid, freq in counts:
            X_spmatrix[did, wid]=freq

    return X_spmatrix.tocsr()


def get_dataset_path(domain_name, exp_type):
    prefix ='./dataset/'
    if exp_type == 'small':
        fname = 'labelled.review'
    elif exp_type == 'all':
        fname = 'all.review'
    elif exp_type == 'test':
        fname = 'unlabeled.review'

    return os.path.join(prefix, domain_name, fname)


def get_dataset(source_name, target_name, max_words=5000):

    source_path  = get_dataset_path(source_name, 'small')
    target_path1 = get_dataset_path(target_name, 'small')
    target_path2 = get_dataset_path(target_name, 'test')

    dataset_list = [source_path, target_path1, target_path2]
    datasets, dico = parse_processed_amazon_dataset(dataset_list, max_words)

    L_s, Y_s = datasets[source_path]
    L_t1, Y_t1 = datasets[target_path1]
    L_t2, Y_t2 = datasets[target_path2]

    X_s  = count_list_to_sparse_matrix(L_s,  dico)
    X_t1 = count_list_to_sparse_matrix(L_t1, dico)
    X_t2 = count_list_to_sparse_matrix(L_t2, dico)

    return X_s, Y_s, X_t1, Y_t1, X_t2, Y_t2, dico