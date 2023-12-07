from simulator import Simulator
import argparse
import random
from deap import base, creator, tools, algorithms
import os
import multiprocessing
import numpy as np
import time

populationSize = 8
numGenerations = 100
mutationRate = 0.35
crossOverRate = 0.8
weightsPerPlayer = 5
logDir = "logs"
winPercentage = {}

# Assuming that your enemy agents' names are valid and will be used for some logic
enemyAgentnames = ["mcts_agent","alex_agent", "emile_agent", "heuristic_agent", "random_agent"]

# The pool creation logic is not fully clear. Assuming you'll select one agent.
chosenAgent = random.choice(enemyAgentnames)

args = argparse.Namespace(
    player_1="student_agent",
    player_2=None,
    player_2_weights=None,
    board_size=None,
    board_size_min=6,
    board_size_max=12,
    display=True,
    display_delay=0.4,
    display_save=False,
    display_save_path="plots/",
    autoplay=True,
    autoplay_runs=10
)

def log_best_weights(best_weights, generation, win_percentages):
    with open(os.path.join(logDir, f"best_weights_gen_{generation}.txt"), 'w') as f:
        f.write(str(best_weights) + "\n")
        f.write("Win Percentages:\n")
        for agent, win_perc in win_percentages.items():
            f.write(f"{agent}: {win_perc*100}%\n")

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def evaluate(individual):
    player_1_weights = individual[:weightsPerPlayer]
    total_score = 0
    win_percentages = {}

    for enemy_agent in enemyAgentnames:
        local_args = argparse.Namespace(**vars(args))
        local_args.player_1_weights = player_1_weights
        local_args.player_2 = enemy_agent
        p1_win_count, p2_win_count = Simulator(local_args).autoplay()
        win_percentage = p1_win_count / args.autoplay_runs  # Assuming autoplay_runs is the total number of games
        win_percentages[enemy_agent] = win_percentage
        total_score += p1_win_count - p2_win_count

    individual_key = str(individual)  # or any other unique identifier
    winPercentage[individual_key] = win_percentages

    return total_score

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=weightsPerPlayer)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=mutationRate)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    if not os.path.exists(logDir):
        os.makedirs(logDir)

    num_processes = 10
    pool = multiprocessing.Pool(processes=num_processes)
    toolbox.register("map", pool.map)

    # Initialize population
    population = toolbox.population(n=populationSize)
    historical_best = []

    for gen in range(numGenerations):
        offspring = algorithms.varAnd(population, toolbox, cxpb=crossOverRate, mutpb=mutationRate)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))

        # Log best weights
        best_ind = tools.selBest(population, 1)[0]
        individual_key = str(best_ind)
        best_win_percentages = winPercentage.get(individual_key, {})
        log_best_weights(best_ind, gen, best_win_percentages) 

        # Store a historical best
        if gen % 10 == 0:
            historical_best.append(best_ind)
        
        # Introduce a random individual to maintain diversity
        if gen % 25 == 0 and historical_best:
            num_to_replace = min(2, len(historical_best))
            for i in range(num_to_replace):
                population[random.randint(0, len(population) - 1)] = toolbox.clone(historical_best[i])

    # Final output
    final_best_ind = tools.selBest(population, 1)[0]
    print("Final Best Individual: %s, %s" % (final_best_ind, final_best_ind.fitness.values))
    log_best_weights(final_best_ind, "final")

    pool.close()
    pool.join()

if __name__ == '__main__':
    time_taken = 0
    main()