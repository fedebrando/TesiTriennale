
# main
MAX_ITERATIONS = 25 # number of max coevolution iterations
DEFAULT_SEED = 3612 # default seed for random function initializing (if not specified)
SAVE_THE_FIRST = False # the first random embedding tree will be saved with the invalid validation score -1 (True -> classifier before embedding, False -> viceversa)
P = 0.6 # portion (from 0 to 1) of population to maintain between one evolution and the next (both classifier and embedding), always including the best-so-far individual to guarantee elitism
MINIMIZE = True # the fitness function must be minimized (both classifier and embedding ones)
NUM_WORSE_VALID_TERMINATION = 6 # number of successive validations without improvement to terminate the coevolution process
RESTART_NON_IMPROVEMENT_CNT = False # restart to 0 the non-improvement-count between two phases
START_EMBEDDING_RANDOM = True # True if start embedding must be randomly choosen from trees which contain all input words 'wi', False if it must be 'w0'

# dataset sectioning
DATASET_PATH = '../data/dataset32.dat'
NUM_CLASS = 10 # number of class (enumerated from 0 to NUM_CLASS-1)
TRAINING_DATA_PERCENTAGE = 0.8 # dataset section (from 0 to 1) for training data (training set + validation set)
VALIDATION_SET_PERCENTAGE = 0.2 # training data section for validation (from 0 to 1)

# inputs
WORD_BIT = 32 # feature bit number
DIM_PATTERN = 4 # number of components for each pattern in dataset (excluding label)

# classifier
CL_FG_MIN_TREE_HEIGHT = 1 # only for the first generation
CL_FG_MAX_TREE_HEIGHT = 7 # only for the first generation
CL_TOURN_SIZE = 7 # multisubset population size in tournament selection
CL_MUT_APP_MIN_TREE_HEIGHT = 1 # max height of appendable tree in mutation
CL_MUT_APP_MAX_TREE_HEIGHT = 6 # min height of appendable tree in mutation
CL_CX_MAX_TREE_HEIGHT = 12 # max reachable height during crossover
CL_MUT_MAX_TREE_HEIGHT = 12 # max reachable height during mutation
CL_CX_MAX_POP_SIZE = 4096 # max reachable population size during crossover
CL_MUT_MAX_POP_SIZE = 4096 # max reachable population size during mutation
CL_WEIGHT_IND_LENGTH_PENALTY = 0.000001 # individual length penalty weight (constrained problem)
CL_INI_POP_SIZE = 1000 # initial population size
CL_CX_PB = 0.8 # crossover probability
CL_MUT_PB = 0.15 # mutation probability
CL_GEN = 20 # max number of generations to evolve a label classifier
CL_GEN_PER_VALIDATION = 3 # number of generations after which a validation takes place
CL_HOF_MAX_SIZE = 1 # hall-of-fame max size (however, also for different value, we'll evaluate only the first best individual)

# embedding
EMB_FG_MIN_TREE_HEIGHT = 1 # only for the first generation
EMB_FG_MAX_TREE_HEIGHT = 7 # only for the first generation
EMB_TOURN_SIZE = 7 # multisubset population size in tournament selection
EMB_MUT_APP_MIN_TREE_HEIGHT = 1 # max height of appendable tree in mutation
EMB_MUT_APP_MAX_TREE_HEIGHT = 6 # min height of appendable tree in mutation
EMB_CX_MAX_TREE_HEIGHT = 12 # max reachable height during crossover
EMB_MUT_MAX_TREE_HEIGHT = 12 # max reachable height during mutation
EMB_CX_MAX_POP_SIZE = 4096 # max reachable population size during crossover
EMB_MUT_MAX_POP_SIZE = 4096 # max reachable population size during mutation
EMB_WEIGHT_IND_LENGTH_PENALTY = 0.000001 # individual length penalty weight (constrained problem)
EMB_INI_POP_SIZE = 1000 # initial population size
EMB_CX_PB = 0.8 # crossover probability
EMB_MUT_PB = 0.15 # mutation probability
EMB_GEN = 20 # max number of generations to evolve a label classifier
EMB_GEN_PER_VALIDATION = 3 # number of generations after which a validation takes place
EMB_HOF_MAX_SIZE = 1 # hall-of-fame max size (however, also for different value, we'll evaluate only the first best individual)
