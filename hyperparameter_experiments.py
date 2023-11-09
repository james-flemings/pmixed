import subprocess
import csv
import tqdm
import numpy as np
import math

results_file = "results.csv"
parameters = ["epsilon", 'num_ensemble', 'query_budget', 'alpha', 'p', 'ppl']

epsilons = [2, 4, 6, 8, 10]
num_ensembles = [8, 16, 32, 64]
query_budgets = [512, 1024, 2048, 4096]
iters = [2**4, 2**3, 2**2, 2**1]
alphas = [3, 4, 5, 6, 7]
p_s = [1/32, 1/16, 1/8, 1/4]

default_epsilon = 8
default_query_budget = 1024 
default_num_ensembles = 32 
default_p = 1/32
default_iters = 2**3
default_alpha = 3
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

def get_results(epsilon, query_budget, alpha, num_ensembles, p, iters):
    command = create_command(epsilon, query_budget, alpha,
                                num_ensembles, p, iters)
    output = subprocess.run(command, stdout=subprocess.PIPE)
    ppl = output.stdout.decode('utf-8').split('\n')[-2].split(" ")[-1]
    results = update_results(epsilon, query_budget, alpha,
                            num_ensembles, p, ppl) 
    return results

with open(results_file, 'w') as f:
    w = csv.DictWriter(f, fieldnames=parameters)
    w.writeheader()

    for epsilon in tqdm.tqdm(epsilons, desc="Epsilon"):
        default_alpha = math.ceil(4 * np.log(1 / delta) / (3 * epsilon) + 1)
        results = get_results(epsilon, default_query_budget, default_alpha, default_num_ensembles, default_p, default_iters)
        w.writerow(results)

    for q_budget, iter in tqdm.tqdm(zip(query_budgets, iters), desc="Query Budget"):
        results = get_results(default_epsilon, q_budget, default_alpha, default_num_ensembles, default_p, iter)
        w.writerow(results)

    default_alpha = 3

    for ensemble in tqdm.tqdm(num_ensembles, desc="Ensemble"):
        default_p = 1/ensemble
        results = get_results(default_epsilon, default_query_budget, default_alpha, ensemble, default_p, default_iters)
        w.writerow(results)

    default_p = 1/32

    for alpha in tqdm.tqdm(alphas, desc="alpha"):
        results = get_results(default_epsilon, default_query_budget, alpha, default_num_ensembles, default_p, default_iters)
        w.writerow(results)

    for p in tqdm.tqdm(p_s, desc="Sample probability"):
        results = get_results(default_epsilon, default_query_budget, default_alpha, default_num_ensembles, p, default_iters)
        w.writerow(results)