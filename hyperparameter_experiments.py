import subprocess
import csv
import tqdm
import numpy as np
import math

results_file = "results.csv"
parameters = ["epsilon", 'num_ensemble', 'query_budget', 'alpha', 'p', 'ppl']

epsilons = [2, 4, 6, 8, 10]
num_ensembles = [8, 16, 32, 64]
query_budgets = [512, 1024, 2048]
alphas = [3, 4, 5, 6, 7]
p_s = [1/16, 1/8, 1/4, 1/2]

default_epsilon = 8
default_query_budget = 512
default_num_ensembles=16
default_p = 1/8
default_iters=5
delta = 1e-5

#results = []

def create_command(epsilon, q_budget, alpha, num_ensemble, p, iters):
    return ['./prediction_experiments.py', f'--epsilon={epsilon}',
               f'--query_budget={q_budget}',
               f'--alpha={alpha}',
               f'--num_ensemble={num_ensemble}',
               f'--p={p}',
               f'--iters={iters}'
    ]

def update_results(epsilon, q_budget, alpha, num_ensemble, p, ppl):
    return {'epsilon': epsilon,
    'query_budget': q_budget,
    'alpha': alpha,
    'num_ensemble': num_ensemble,
    'p': p,
    'ppl': ppl
    }

with open(results_file, 'w') as f:
    w = csv.DictWriter(f, fieldnames=parameters)
    w.writeheader()

    for epsilon in tqdm.tqdm(epsilons, desc="Epsilon"):
        default_alpha = math.ceil(4 * np.log(1 / delta) / (3 * epsilon) + 1)
        command = create_command(epsilon, default_query_budget, default_alpha,
                                default_num_ensembles, default_p, default_iters)
        result = subprocess.run(command, stdout=subprocess.PIPE)
        ppl = result.stdout.decode('utf-8').split('\n')[-2]
        results = update_results(epsilon, default_query_budget, default_alpha,
                                default_num_ensembles, default_p, ppl) 
        w.writerow(results)

    default_alpha = 3

    for ensemble in tqdm.tqdm(num_ensembles, desc="Ensemble"):
        command = create_command(default_epsilon, default_query_budget, default_alpha,
                                ensemble, default_p, default_iters)
        result = subprocess.run(command, stdout=subprocess.PIPE)
        ppl = result.stdout.decode('utf-8').split('\n')[-2]
        results = update_results(default_epsilon, default_query_budget, default_alpha,
                                ensemble, default_p, ppl) 
        w.writerow(results)

    for alpha in tqdm.tqdm(alphas, desc="alpha"):
        command = create_command(default_epsilon, default_query_budget, alpha,
                                default_num_ensembles, default_p, default_iters)
        result = subprocess.run(command, stdout=subprocess.PIPE)
        ppl = result.stdout.decode('utf-8').split('\n')[-2]
        results = update_results(default_epsilon, default_query_budget, alpha,
                                default_num_ensembles, default_p, ppl) 
        w.writerow(results)

    for q_budget in tqdm.tqdm(query_budgets, desc="Query Budget"):
        command = create_command(default_epsilon, q_budget, default_alpha,
                                default_num_ensembles, default_p, default_iters)
        result = subprocess.run(command, stdout=subprocess.PIPE)
        ppl = result.stdout.decode('utf-8').split('\n')[-2]
        results = update_results(default_epsilon, q_budget, default_alpha,
                                default_num_ensembles, default_p, ppl) 
        w.writerow(results)

    for p in tqdm.tqdm(p_s, desc="Sample probability"):
        command = create_command(default_epsilon, default_query_budget, default_alpha,
                                default_num_ensembles, p, default_iters)
        result = subprocess.run(command, stdout=subprocess.PIPE)
        ppl = result.stdout.decode('utf-8').split('\n')[-2]
        results = update_results(default_epsilon, default_query_budget, default_alpha,
                                default_num_ensembles, p, ppl) 
        w.writerow(results)