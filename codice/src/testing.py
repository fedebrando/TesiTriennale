
from _cfg_params import DIM_PATTERN, NUM_CLASS
from dataset_sectioning import read_data_from
from functools import partial
from deap import gp
from classifier import define_primitive_set as cl_pset
from embedding import define_primitive_set as emb_pset
from os.path import isfile
import warnings

def binary_sa_classifiers(label: int) -> list[tuple[int, callable]]: # returns list of couples (label, stand_alone_classifier)
    pset = cl_pset()
    file_path = f'../classifiers/best_%dclassifiers.tr' % label
    if not isfile(file_path): # file doesn't exist
        return []
    best_rows = read_data_from(file_path, False)

    for i in range(len(best_rows)):
        best_rows[i][1] = gp.compile(best_rows[i][0], pset)
        best_rows[i][2] = int(best_rows[i][1])
    return [(row[0], partial((lambda row, *w : (row[0](*w) & 2**row[1]) != 0), row)) for row in best_rows]

def binary_emb_classifier(label: int) -> list[tuple[int, callable]]: # returns list of couples (label, embedding_classifier)
    pset_cl = cl_pset()
    pset_emb = emb_pset()
    emb_path = f'../classifiers/best_%dembeddings.tr' % label
    cl_path = f'../classifiers/best_%dclassifiers.tr' % label
    if not (isfile(emb_path) and isfile(cl_path)): # one file (at least) doesn't exist
        return []
    best_emb = read_data_from(emb_path, False)
    best_cl = read_data_from(cl_path, False)

    classifiers = []
    for i in range(len(best_emb)): # we suppose that len(best_cl) = len(best_emb)
        emb = gp.compile(best_emb[i][0], pset_emb)
        cl = gp.compile(best_cl[i][0], pset_cl)
        best_bit = int(best_cl[i][1])

        bin_sa_cl = partial((lambda we, best_bit, cl : (cl(we) & 2**best_bit) != 0), best_bit=best_bit, cl=cl)
        bin_emb_cl = partial((lambda emb, bin_sa_cl, *w : bin_sa_cl(emb(*w))), emb, bin_sa_cl) # composition from cl and emb
        classifiers.append((label, bin_emb_cl))
    return classifiers

def testing_binary_classifier(test_set: list, bincl: tuple[int, callable]) -> dict: # returns confusion matrix from classifier testing
    tp, tn, fp, fn = 0, 0, 0, 0
    label, bcfunc = bincl

    for features in test_set:
        real = features[DIM_PATTERN] == label
        prediction = bcfunc(*features[:-1])
        if prediction == real:
            if real:
                tp += 1
            else:
                tn += 1
        else:
            if prediction:
                fp += 1
            else:
                fn += 1
    return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}

def tpr(cm: dict) -> float: # true-positive-rate (sensitivity or recall)
    return cm['TP']/(cm['TP'] + cm['FN'])

def tnr(cm: dict) -> float: # true-negative-rate (specificity)
    return cm['TN']/(cm['TN'] + cm['FP'])

def printc(s=''): # printing with end='\t'
    print(s, end='\t')

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning) # to ignore creator.create() equal-names runtime warnings
    test_set = read_data_from('../data/test_set.dat')

    print('label\tlast\tTPR\tTNR')
    for label in range(NUM_CLASS):
        bin_cl = binary_emb_classifier(label)
        for bc in bin_cl:
            confusion_matrix = testing_binary_classifier(test_set, bc)
            printc(label)
            printc('*' if bc == bin_cl[len(bin_cl)-1] else '') # the last one (*)
            printc(round(100*tpr(confusion_matrix), 2))
            print(round(100*tnr(confusion_matrix), 2))
    print()


# entry point
if __name__ == '__main__':
    main()
