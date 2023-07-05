
from _cfg_params import *
import operator as op
import random
import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import gp
from matplotlib import pyplot as plt
from ea_simple import *
import pathos.multiprocessing as pm

# primitive set
def nand(a: int, b: int) -> int:
    return op.__invert__(op.__and__(a, b))

def nor(a: int, b: int) -> int:
    return op.__invert__(op.__or__(a, b))

def lcshf(a: int) -> int: # less circular shift
    la = (a << 1) & (2**WORD_BIT - 1) # we're working with WORD_BIT bits
    return la if a < 2**(WORD_BIT - 1) else la+1

def rcshf(a: int) -> int: # right circular shift
    wa = a >> 1
    return wa if a % 2 == 0 else wa+2**(WORD_BIT - 1)

def circShift(a: int, n: int, right: bool) -> int: # circular n-shift of a
    cshf = rcshf if right else lcshf
    for _ in range(n):
        a = cshf(a)
    return a

def define_primitive_set() -> gp.PrimitiveSet:
    # creator settings
    creator.create("FitnessMin", base.Fitness, weights=(-1.0 if MINIMIZE else +1.0,)) # there is only a fitness to be minimized
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin) # an individual is a tree, but contains its fitness

    pset = gp.PrimitiveSet('pset', 1)

    # terminal set (T)
    pset.renameArguments(ARG0='we') # embedding argument (we = emb(w0, w1, ...))
    #pset.addTerminal(0)
    #pset.addTerminal(1)
    pset.addEphemeralConstant(f'ERC%d_cl' % define_primitive_set.count, lambda : random.randint(0, 2**WORD_BIT - 1)) # ephemeral random constant
    define_primitive_set.count += 1

    # function set (F)
    pset.addPrimitive(op.__and__, 2)
    pset.addPrimitive(op.__or__, 2)
    pset.addPrimitive(op.__invert__, 1)
    pset.addPrimitive(op.__xor__, 2)
    pset.addPrimitive(nand, 2)
    pset.addPrimitive(nor, 2)
    pset.addPrimitive(lcshf, 1)
    pset.addPrimitive(rcshf, 1)
    pset.addPrimitive(lambda a : circShift(a, 2, False), 1, 'lcshf2') # double left circular shift
    pset.addPrimitive(lambda a : circShift(a, 2, True), 1, 'rcshf2')
    pset.addPrimitive(lambda a : circShift(a, 4, False), 1, 'lcshf4')
    pset.addPrimitive(lambda a : circShift(a, 4, True), 1, 'rcshf4')

    return pset
define_primitive_set.count = 0

def bit_fitness(ind_length: int, fp: int, fn: int, totp: int, totn: int) -> float: # return bit fitness value
    return np.sqrt(50*(fp**2 + fn**2) / (totp + totn)**2) + ind_length*CL_WEIGHT_IND_LENGTH_PENALTY # the second adding is the length-proportional individual penalty

def fitness_bestbit_for(label_classifier: int, individual: gp.PrimitiveTree, evaluation_set: list, compile: callable) -> tuple[float, int]: # fitness function and the best bit for all classifier
    confusion_matrix = {}
    confusion_matrix['TP'] = [0] * WORD_BIT # it isn't a bit, but a count value (true positive)
    confusion_matrix['TN'] = [0] * WORD_BIT # true negative
    confusion_matrix['FP'] = [0] * WORD_BIT # false positive
    confusion_matrix['FN'] = [0] * WORD_BIT # false negative
    tot_P, tot_N = 0, 0 # total of positives and negatives

    func_ind = compile(individual)
    for features in evaluation_set:
        result = func_ind(we=features[0])
        positive = (features[1] == label_classifier) # features[1] is the label of the input features[0]
        if positive:
            tot_P += 1
        else:
            tot_N += 1
        bitmask = 1 # 000...001b, 64 bits
        for j in range(WORD_BIT):
            bit_j = result & bitmask
            str_case = ('TP' if bit_j else 'FN') if positive else ('FP' if bit_j else 'TN')
            confusion_matrix[str_case][j] += 1
            bitmask <<= 1
    bit_fitness_lst = [bit_fitness(len(individual), confusion_matrix["FP"][j], confusion_matrix["FN"][j], tot_P, tot_N) for j in range(WORD_BIT)]
    min_bit_fitness = min(bit_fitness_lst) # final fitness value is the minimum among all bit fitness values

    return min_bit_fitness, bit_fitness_lst.index(min_bit_fitness) 

def fitness_for(label_classifier: int, individual: gp.PrimitiveTree, evaluation_set: list, compile: callable) -> float: # only fitness function for all classifier (to minimize)
    return fitness_bestbit_for(label_classifier, individual, evaluation_set, compile)[0] # (fitness, best_bit)[0] -> only fitness

# GP parameters
def define_gp_params(label_classifier: int, pset: gp.PrimitiveSet, training_set: list) -> base.Toolbox: # toolbox creation
    toolbox = base.Toolbox()

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=CL_FG_MIN_TREE_HEIGHT, max_=CL_FG_MAX_TREE_HEIGHT) # some kind of gene (allele)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr) # individual (a syntax tree)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual) # population
    toolbox.register("compile", gp.compile, pset=pset) # tree compilation -> a program (a function)

    toolbox.register("evaluate", lambda individual : (fitness_for(label_classifier, individual, training_set, toolbox.compile),)) # fitness function
    toolbox.register("select", tools.selTournament, tournsize=CL_TOURN_SIZE) # selection (tournament selection)

    toolbox.register("mate", gp.cxOnePoint) # crossover (one-point cx)

    toolbox.register("expr_mut", gp.genFull, min_=CL_MUT_APP_MIN_TREE_HEIGHT, max_=CL_MUT_APP_MAX_TREE_HEIGHT) # subtree to append in mutation
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset) # mutation (subtree mutation)

    toolbox.decorate("mate", gp.staticLimit(key=op.attrgetter("height"), max_value=CL_CX_MAX_TREE_HEIGHT)) # tree height limits (bloat)
    toolbox.decorate("mutate", gp.staticLimit(key=op.attrgetter("height"), max_value=CL_MUT_MAX_TREE_HEIGHT))

    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=CL_CX_MAX_POP_SIZE)) # pop size limits
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=CL_MUT_MAX_POP_SIZE))

    #toolbox.register("map", pm.Pool().map) # map multiprocessing

    return toolbox

def define_statistics() -> tools.Statistics:
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    return stats

def plot_statistics(label_classifier: int, logbook: tools.Logbook, valid_dict: dict, save_not_show: bool=False, seed=None):
    # plot font settings
    plt.rcParams['font.sans-serif'] = "Latin Modern Math" # according to LatEX style
    plt.rcParams['font.family'] = "sans-serif"

    gen = logbook.select("gen")
    min = logbook.select("min")
    avg = logbook.select("avg")
    fig, ax1 = plt.subplots()

    fig.suptitle((f'Classifier for label %d' % label_classifier) + ((f' (seed %d)' % seed) if seed else ''))
    line1 = ax1.plot(gen, min, "b-", label="Min Fitness")
    line3 = ax1.plot(valid_dict["gen"], valid_dict["valid"], "b--", label="Validation score")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax2 = ax1.twinx()
    line2 = ax2.plot(gen, avg, "r-", label="Average Fitness")
    ax2.set_ylabel("", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    lns = line1 + line2 + line3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")

    if save_not_show:
        plt.savefig((f'../classifiers/last_%dclassifier' % label_classifier) + ((f'_seed%d' % seed) if seed else '') + '.svg')
    else:
        plt.show()

def evaluate_validation_score(label_classifier: int, best_so_far: gp.PrimitiveTree, validation_set: list, compile: callable) -> tuple[float, int]: # best-so-far tree validation measure (it is based on fitness: the lower the better) and its best bit (with that fitness)
    return fitness_bestbit_for(label_classifier, best_so_far, validation_set, compile)

# single-classifier evolution process (returns hall-of-fame, validation score for hof[0], the best bit for hof[0], logbook, validation dictionary, the last population and non-improvement-count)
def evolution(label_classifier: int, training_set: list, validation_set: list, pset: gp.PrimitiveSet, survived_pop: list, last_nic: int, best_valid_score: float) -> tuple[tools.HallOfFame, int, float, tools.Logbook, dict, list]: 
    toolbox = define_gp_params(label_classifier, pset, training_set)
    ini_pop = survived_pop + toolbox.population(n=CL_INI_POP_SIZE-len(survived_pop))
    hof = tools.HallOfFame(CL_HOF_MAX_SIZE)
    gen_count = 0
    partial_end_pop = ini_pop
    stats = define_statistics()
    logbook = tools.Logbook()

    best_so_far = (None, best_valid_score, None) if best_valid_score else None
    not_improvement_count = 0 if RESTART_NON_IMPROVEMENT_CNT else last_nic
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    valid_dict = {'gen': [], 'valid': []} # a kind of validation logbook
    while gen_count < CL_GEN and not_improvement_count < NUM_WORSE_VALID_TERMINATION: # actually, != is enough (instead of <, for both)
        gen = CL_GEN_PER_VALIDATION if gen_count + CL_GEN_PER_VALIDATION <= CL_GEN else CL_GEN - gen_count
        partial_end_pop = eaSimple(partial_end_pop, toolbox, CL_CX_PB, CL_MUT_PB, gen, stats, logbook, halloffame=hof, verbose=True)
        gen_count += gen
        
        if (not best_so_far) or hof[0] != best_so_far[0]: # evaluate validation score only if it is necessary
            valid_score = evaluate_validation_score(label_classifier, hof[0], validation_set, toolbox.compile)
        if not best_so_far:
            best_so_far = (hof[0], valid_score[0], valid_score[1])
        elif valid_score[0] < best_so_far[1]:
            best_so_far = (hof[0], valid_score[0], valid_score[1])
            not_improvement_count = 0
        else:
            not_improvement_count += 1
        print("validation score: " + str(valid_score) + "\t[non-improvements cnt: " + str(not_improvement_count) + "]")
        valid_dict["gen"].append(gen_count)
        valid_dict["valid"].append(valid_score[0])

    return hof, best_so_far[1], best_so_far[2], logbook, valid_dict, partial_end_pop, not_improvement_count

def save_best_of(label_classifier: int, best: gp.PrimitiveTree, best_bit: int, valid_score: float):
    with open(f'../classifiers/best_%dclassifiers.tr' % label_classifier, mode='a') as fbests:
        fbests.write(';'.join([str(best), str(best_bit), str(valid_score)]) + '\n')
        fbests.close()

# returns the best-so-far primitive tree, the best bit, the best fitness (validation score) value (of the same individual), survival population and non-improvement-count
def clmain(label_classifier: int, transformed_training: list, transformed_validation: list, survived_pop: list, last_nic: int, best_valid_score: float) -> tuple[gp.PrimitiveTree, int, float, list]:
    pset = define_primitive_set()
    print(f"phase C: start for %d-label classifier" % label_classifier) # start log
    hof, valid_score, best_bit, logbook, valid_dict, pop, nic = evolution(label_classifier, transformed_training, transformed_validation, pset, survived_pop, last_nic, best_valid_score) # evolution
    plot_statistics(label_classifier, logbook, valid_dict, save_not_show=True) # save the statistic graphs
    save_best_of(label_classifier, hof[0], best_bit, valid_score) # save the best-so-far tree
    print(f"saving for %d-digit classifier (end)" % label_classifier) # end log

    return gp.compile(hof[0], pset), best_bit, valid_score, survival_pop(pop, hof[0], CL_INI_POP_SIZE), nic
            