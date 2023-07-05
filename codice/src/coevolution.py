
from _cfg_params import DEFAULT_SEED, MAX_ITERATIONS, NUM_WORSE_VALID_TERMINATION
from classifier import clmain
from embedding import embmain, data_transformation
from dataset_sectioning import read_data_from
from functools import partial
import random
import warnings
import sys

def check_args(argv: list) -> bool: # checks argv and prints any error logs
    if not (len(argv) in [2, 3]):
        print(f'Error. Usage: %s label [seed]' % argv[0])
        return False
    if any([not arg.isnumeric() for arg in argv[1:]]):
        print("Error. Consider that 'label' and 'seed' must be natural numbers.")
        return False
    return True

def early_exit_log():
    print('--- Early Exit ---')

def main(argv: list): # coevolution label [seed]
    if not check_args(argv):
        return
    
    label = int(argv[1])
    random.seed(int(argv[2] if len(argv) == 2+1 else DEFAULT_SEED)) # random seed initializing
    warnings.filterwarnings("ignore", category=RuntimeWarning) # to ignore creator.create() equal-names runtime warnings
    training_set = read_data_from('../data/training_set.dat')
    validation_set = read_data_from('../data/validation_set.dat')
    bsf_bin_classifier = (label, None)
    emb_survival, cl_survival = [], []
    nic = 0 # non-improvement-count
    emb_valid_score, cl_valid_score = None, None
    for iter in range(MAX_ITERATIONS):
        print(f'--- Iter %d ---' % iter) # start iter log

        # evolving embedding (with fixed classifier, after first embedding) - phase E
        embedding, emb_valid_score, emb_survival, nic = embmain(training_set, validation_set, bsf_bin_classifier, emb_survival, nic, emb_valid_score)
        if nic == NUM_WORSE_VALID_TERMINATION:
            early_exit_log()
            break

        # data transformation
        trans_training = data_transformation(training_set, embedding)
        trans_validation = data_transformation(validation_set, embedding)

        # evolving binary classifier (with fixed embedding) - phase C
        bsf_func, best_bit, cl_valid_score, cl_survival, nic = clmain(label, trans_training, trans_validation, cl_survival, nic, cl_valid_score)
        if nic == NUM_WORSE_VALID_TERMINATION:
            early_exit_log()
            break
        bsf_bin_classifier = label, partial((lambda we, bsf_func, best_bit : (bsf_func(we) & 2**best_bit) != 0), bsf_func=bsf_func, best_bit=best_bit)


# entry point
if __name__ == '__main__':
    main(sys.argv)
