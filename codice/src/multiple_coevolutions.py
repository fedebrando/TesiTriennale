
from _cfg_params import NUM_CLASS
from multiprocessing import Pool
import subprocess as sp
import dataset_sectioning
import testing

def call_coevolution(args: tuple[int, int]): # run coevolution.py module with received in-line arguments
    label, seed = args
    cmd = ['python', 'coevolution.py', str(label)]
    if seed:
        cmd.append(str(seed))
    sp.call(cmd)
    
def main():
    dataset_sectioning.main() # dataset -> training, validation, test
    with Pool() as pool: # a process for each label coevolution
        pool.map(call_coevolution, [(label, None) for label in [0, 5, 6, 7, 8, 9]]) # None -> DEFAULT_SEED
    testing.main() # testing


# entry point
if __name__ == '__main__':
    main()
