
from _cfg_params import P, MINIMIZE
from deap import algorithms as algo
from deap import gp
from random import randint

# deap eaSimple evolution algorithm with output and logbook modifies and start-generation extension (returns end population)
def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats, logbook, halloffame, verbose=__debug__) -> list:
    genstart = len(logbook) # start generation
    best_so_far = None

    # evaluate the individuals with an invalid fitness
    if genstart == 0:
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        halloffame.update(population)
        best_so_far = halloffame[0]

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    # begin the generational process
    gen_range = range(1, ngen + 1) if genstart == 0 else range(genstart, genstart + ngen)
    for gen in gen_range:
        # select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # vary the pool of individuals (crossover and mutation)
        offspring = algo.varAnd(offspring, toolbox, cxpb, mutpb)

        # evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # update the hall of fame with the generated individuals
        if not best_so_far:
            best_so_far = halloffame[0]
        elif (+1 if MINIMIZE else -1)*(best_so_far.fitness.values[0] - halloffame[0].fitness.values[0]) > 0:
            best_so_far = halloffame[0] # best-so-far individual in last population
        halloffame.update(offspring)

        # replace the current population by the offspring
        population[:] = offspring

        # elitism
        if best_so_far not in population:
            population[randint(0, len(population)-1)] = best_so_far

        # append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population

def survival_pop(pop: list, bsf: gp.PrimitiveTree, ini_pop_size: int) -> list: # survived population (for elitism)
    len_quartile = int(ini_pop_size/4)
    prev_len_sur_pop = int(P*ini_pop_size) # +-1
    if prev_len_sur_pop == 0:
        return [bsf]
    pop.sort(key=(lambda ind : ind.fitness.values[0]), reverse=(not MINIMIZE))
    sur_pop = []
    for i in range(4): # four quartiles
        start = i*len_quartile
        prob_end = start + int(((4-i)/10)*prev_len_sur_pop) # but it could be out of the quartile...
        end = (prob_end if prob_end <= start + len_quartile else start + len_quartile) # ...we must control that
        if end > ini_pop_size:
            end = ini_pop_size
        sur_pop += pop[start : end] # 40% from the first section (P) quartile, 30% from the second one ecc... (tot: 100%)
    len_sur_pop = len(sur_pop)

    if bsf not in sur_pop: # best-so-far individual must survive (elitism)
        if prev_len_sur_pop < len_sur_pop:
            sur_pop.append(bsf)
        else:
            sur_pop[randint(0, len_sur_pop-1)] = bsf

    return sur_pop
