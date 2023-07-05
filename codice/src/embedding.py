
from _cfg_params import *
import operator as op
import random
import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import gp
from matplotlib import pyplot as plt
from typing import Union
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

    # terminal set (T)
    pset = gp.PrimitiveSet('pset', DIM_PATTERN, prefix='w') # 64-bit input words
    #pset.addTerminal(0)
    #pset.addTerminal(1)
    pset.addEphemeralConstant(f'ERC%d_emb' % define_primitive_set.count, lambda : random.randint(0, 2**WORD_BIT - 1)) # ephemeral random constant
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

def data_transformation(data_before: list, t_func: callable): # returns before_data transformed by t_func
    return [[t_func(*feat[:-1]), feat[DIM_PATTERN]] for feat in data_before]

def bit_fitness(ind_length: int, fp: int, fn: int, totp: int, totn: int) -> float: # return bit fitness value
    return np.sqrt(50*(fp**2 + fn**2) / (totp + totn)**2) + ind_length*EMB_WEIGHT_IND_LENGTH_PENALTY # the second adding is the length-proportional individual penalty

def fitness_for(individual: gp.PrimitiveTree, evaluation_set: list, bsf_bin_cl: tuple[int, callable], compile: callable) -> tuple[float, int]: # fitness function and the best bit for all classifier
    trans_eval_set = data_transformation(evaluation_set, compile(individual))
    label, bin_cl_func = bsf_bin_cl
    fp, fn, totp, totn = 0, 0, 0, 0

    for features in trans_eval_set:
        real = features[1] == label
        prediction = bin_cl_func(features[0])
        if real:
            totp += 1
            if not prediction:
                fn += 1
        else:
            totn += 1
            if prediction:
                fp += 1
    return bit_fitness(len(individual), fp, fn, totp, totn)

# GP parameters
def define_gp_params(pset: gp.PrimitiveSet, training_set: list, bsf_bin_cl: tuple[int, callable]) -> base.Toolbox: # toolbox creation
    toolbox = base.Toolbox()

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=EMB_FG_MIN_TREE_HEIGHT, max_=EMB_FG_MAX_TREE_HEIGHT) # some kind of gene (allele)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr) # individual (a syntax tree)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual) # population
    toolbox.register("compile", gp.compile, pset=pset) # tree compilation -> a program (a function)

    toolbox.register("evaluate", lambda individual : (fitness_for(individual, training_set, bsf_bin_cl, toolbox.compile),)) # fitness function
    toolbox.register("select", tools.selTournament, tournsize=EMB_TOURN_SIZE) # selection (tournament selection)

    toolbox.register("mate", gp.cxOnePoint) # crossover (one-point cx)

    toolbox.register("expr_mut", gp.genFull, min_=EMB_MUT_APP_MIN_TREE_HEIGHT, max_=EMB_MUT_APP_MAX_TREE_HEIGHT) # subtree to append in mutation
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset) # mutation (subtree mutation)

    toolbox.decorate("mate", gp.staticLimit(key=op.attrgetter("height"), max_value=EMB_CX_MAX_TREE_HEIGHT)) # tree height limits
    toolbox.decorate("mutate", gp.staticLimit(key=op.attrgetter("height"), max_value=EMB_MUT_MAX_TREE_HEIGHT))

    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=EMB_CX_MAX_POP_SIZE)) # pop size limits (bloat)
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=EMB_MUT_MAX_POP_SIZE))

    #toolbox.register("map", pm.Pool().map) # map multiprocessing

    return toolbox

def define_statistics() -> tools.Statistics:
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    return stats

def plot_statistics(label_embedding: int, logbook: tools.Logbook, valid_dict: dict, save_not_show: bool=False, seed=None):
    # plot font settings
    plt.rcParams['font.sans-serif'] = "Latin Modern Math" # according to LatEX style
    plt.rcParams['font.family'] = "sans-serif"

    gen = logbook.select("gen")
    min = logbook.select("min")
    avg = logbook.select("avg")
    fig, ax1 = plt.subplots()

    fig.suptitle((f'Embedding for label %d' % label_embedding) + ((f' (seed %d)' % seed) if seed else ''))
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
        plt.savefig((f'../classifiers/last_%dembedding' % label_embedding) + ((f'_seed%d' % seed) if seed else '') + '.svg')
    else:
        plt.show()

def evaluate_validation_score(best_so_far: gp.PrimitiveTree, validation_set: list, bsf_bin_cl: tuple[int, callable], compile: callable) -> tuple[float, int]: # best-so-far tree validation measure (it is based on fitness: the lower the better) and its best bit (with that fitness)
    return fitness_for(best_so_far, validation_set, bsf_bin_cl, compile)

def is_acceptable(tree: gp.PrimitiveTree) -> bool: # returns True if tree contains all input words w0, w1, ... w(DIM_PATTERN-1)
    tree_str = str(tree)
    acceptable = True
    for i in range(DIM_PATTERN):
        acceptable = acceptable and (f'w%d' % i) in tree_str
        if not acceptable:
            return False
    return True

def acceptable_embedding_trees(tree_lst: list) -> list:  # only acceptable trees
    return [tree for tree in tree_lst if is_acceptable(tree)]

# start embedding
def start_embedding(pset: gp.PrimitiveSet, toolbox: base.Toolbox) -> gp.PrimitiveTree:
    if not START_EMBEDDING_RANDOM:
        return creator.Individual.from_string("w0", pset) # the start embedding is the first word (only-root tree)
    ini_pop = toolbox.population(n=EMB_INI_POP_SIZE)
    acceptable_trees = acceptable_embedding_trees(ini_pop)
    while (len_ac := len(acceptable_trees)) == 0: # is empty
        ini_pop = toolbox.population(n=EMB_INI_POP_SIZE)
        acceptable_trees = acceptable_embedding_trees(ini_pop)
    return acceptable_trees[random.randint(0, len_ac-1)]

# single-classifier evolution process (returns hall-of-fame, validation score for hof[0], the best bit for hof[0], logbook, validation dictionary and non-improvement-count OR random init indivividual if bsf_bin_cl is None)
def evolution(training_set: list, validation_set: list, pset: gp.PrimitiveSet, bsf_bin_cl: tuple[int, callable], survived_pop: list, last_nic: int, best_valid_score: float) -> Union[tuple[tools.HallOfFame, int, float, tools.Logbook, dict, int], gp.PrimitiveTree]:
    toolbox = define_gp_params(pset, training_set, bsf_bin_cl)
    if not bsf_bin_cl[1]: # there are no classifiers
        return start_embedding(pset, toolbox)
    ini_pop = survived_pop + toolbox.population(n=EMB_INI_POP_SIZE-len(survived_pop))
    
    hof = tools.HallOfFame(EMB_HOF_MAX_SIZE)
    gen_count = 0
    partial_end_pop = ini_pop
    stats = define_statistics()
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    best_so_far = (None, best_valid_score) if best_valid_score else None
    not_improvement_count = 0 if RESTART_NON_IMPROVEMENT_CNT else last_nic
    valid_dict = {'gen': [], 'valid': []} # a kind of validation logbook
    while gen_count < EMB_GEN and not_improvement_count < NUM_WORSE_VALID_TERMINATION: # actually, != is enough (instead of <, for both)
        gen = EMB_GEN_PER_VALIDATION if gen_count + EMB_GEN_PER_VALIDATION <= EMB_GEN else EMB_GEN - gen_count
        partial_end_pop = eaSimple(partial_end_pop, toolbox, EMB_CX_PB, EMB_MUT_PB, gen, stats, logbook, halloffame=hof, verbose=True)
        gen_count += gen

        if (not best_so_far) or hof[0] != best_so_far[0]: # evaluate validation score only if it is necessary
            valid_score = evaluate_validation_score(hof[0], validation_set, bsf_bin_cl, toolbox.compile)
        if not best_so_far:
            best_so_far = (hof[0], valid_score)
        elif valid_score < best_so_far[1]:
            best_so_far = (hof[0], valid_score)
            not_improvement_count = 0
        else:
            not_improvement_count += 1
        print("validation score: " + str(valid_score) + "\t[non-improvements cnt: " + str(not_improvement_count) + "]")
        valid_dict["gen"].append(gen_count)
        valid_dict["valid"].append(valid_score)

    return hof, best_so_far[1], logbook, valid_dict, partial_end_pop, not_improvement_count

def save_best_of(label_embedding: int, best: gp.PrimitiveTree, valid_score: float):
    with open(f'../classifiers/best_%dembeddings.tr' % label_embedding, mode='a') as fbests:
        fbests.write(';'.join([str(best), str(valid_score)]) + '\n')
        fbests.close()

# returns the best-so-far embedding tree individual, its validation score (if bsf_bin_classifier is None, then it returns a random init primitive tree individual), survival population and non-improvement-count
def embmain(training_set: list, validation_set: list, bsf_bin_classifier: tuple[int, callable], survived_pop: list, last_nic: int, best_valid_score: float) -> tuple[callable, float, list, int]:
    label_embedding = bsf_bin_classifier[0]
    pset = define_primitive_set()
    print((f"phase E: start for %d-label embedding" % label_embedding) if bsf_bin_classifier[1] else 'phase E0: start first embedding') # start log
    results = evolution(training_set, validation_set, pset, bsf_bin_classifier, survived_pop, last_nic, best_valid_score) # evolution
    if not bsf_bin_classifier[1]:
        if SAVE_THE_FIRST:
            save_best_of(label_embedding, results, -1) # save start embedding tree (-1 is not significant for validation score)
        return gp.compile(results, pset), best_valid_score, [], last_nic # (result, without s)
    else:
        hof, valid_score, logbook, valid_dict, pop, nic = results
        plot_statistics(label_embedding, logbook, valid_dict, save_not_show=True) # save the statistic graphs
        save_best_of(label_embedding, hof[0], valid_score) # save the best-so-far tree
        print(f"saving for %d-label embedding (end)" % label_embedding) # end log
        return gp.compile(hof[0], pset), valid_score, survival_pop(pop, hof[0], EMB_INI_POP_SIZE), nic
            