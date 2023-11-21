import matplotlib.pyplot as plt 
import csv
import pandas as pd
import numpy as np

eps_s, eps_e = 0, 5 
qb_s, qb_e = 5, 9 
ens_s, ens_e = 9, 12 
alp_s, alp_e = 12, 16 
p_s, p_e = 16, 20

pre_trained = 40.86
fine_tuned = 27.25
yticks = [25, 30, 35, 40]

width = 0.2
data = {"Pre-Trained": [38.63, 67.93], "Fine-Tuned": [25.11, 40.93],
        "Sample-Level\nDP-SGD": [32.96, 53.95], "PMixED": [31.95, 50.91]}

datasets = ["WikiText-103", "One Billion Word"]
x = np.arange(len(datasets))

pre_trained_bars = plt.bar(x-3*width/2, data['Pre-Trained'], width, label="Pre-Trained", color='green')
fine_tuned_bars = plt.bar(x-width/2, data['Fine-Tuned'], width, label="Fine-Tuned", color='red')
dpsgd_bars = plt.bar(x + width/2, data["Sample-Level\nDP-SGD"], width, label="Sample-Level\nDP-SGD", facecolor='deepskyblue', hatch='/')
pmixed_bars = plt.bar(x + 3*width/2, data['PMixED'], width, label="PMixED", color='gold')

for bars in [pre_trained_bars, fine_tuned_bars, dpsgd_bars, pmixed_bars]:
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x()+0.1, yval+0.1, yval, ha='center')

plt.legend()
plt.xticks(x, datasets)
plt.ylabel("PPL")
plt.savefig("plts/comparison.png")
plt.clf()

file_name = "results.csv"
df = pd.read_csv(file_name)

plt.plot(df.iloc[eps_s:eps_e]['epsilon'], df.iloc[eps_s:eps_e]['ppl'], linewidth=2,
          label="PMixED", color='gold', marker='o')
plt.plot(df.iloc[eps_s:eps_e]['epsilon'], [pre_trained for _ in range((eps_e-eps_s))],
          linestyle='dashed', linewidth=2, label='Pre-trained', color="green")
plt.plot(df.iloc[eps_s:eps_e]['epsilon'], [fine_tuned for _ in range((eps_e-eps_s))],
          linestyle='dashed', linewidth=2, label='Fine-tuned', color="red")
plt.xlabel("Epsilon")
plt.yticks(yticks)
plt.xticks([2, 4, 6, 8, 10])
plt.ylabel("PPL")
plt.legend(loc='center left')
plt.savefig('plts/epsilon.png')
plt.clf()

plt.plot(df.iloc[ens_s:ens_e]['num_ensemble'].astype(str), df.iloc[ens_s:ens_e]['ppl'],
          linewidth=2, label="PMixED", color='gold', marker='o')
plt.plot(df.iloc[ens_s:ens_e]['num_ensemble'].astype(str), [pre_trained for _ in range((ens_e-ens_s))],
          linestyle='dashed', linewidth=2, label='Pre-trained', color="green")
plt.plot(df.iloc[ens_s:ens_e]['num_ensemble'].astype(str), [fine_tuned for _ in range((ens_e-ens_s))],
          linestyle='dashed', linewidth=2, label='Fine-tuned', color="red")
plt.xlabel("Number of Ensembles")
plt.yticks(yticks)
#plt.xticks([8, 16, 32, 64])
plt.ylabel("PPL")
plt.legend(loc='upper left', bbox_to_anchor=(0, 0.9))
plt.savefig('plts/ensemble.png')
plt.clf()

plt.plot(df.iloc[alp_s:alp_e]['alpha'], df.iloc[alp_s:alp_e]['ppl'],
          linewidth=2, label="PMixED", color='gold', marker='o')
plt.plot(df.iloc[alp_s:alp_e]['alpha'], [pre_trained for _ in range((alp_e-alp_s))],
          linestyle='dashed', linewidth=2, label='Pre-trained', color="green")
plt.plot(df.iloc[alp_s:alp_e]['alpha'], [fine_tuned for _ in range((alp_e-alp_s))],
          linestyle='dashed', linewidth=2, label='Fine-tuned', color="red")
plt.xlabel("Alpha")
plt.yticks(yticks)
plt.xticks([3, 4, 5, 6])
plt.ylabel("PPL")
plt.legend()
plt.savefig('plts/alpha.png')
plt.clf()

plt.plot(df.iloc[qb_s:qb_e]['query_budget'], df.iloc[qb_s:qb_e]['ppl'],
          linewidth=2, label="PMixED", color='gold', marker='o')
plt.plot(df.iloc[qb_s:qb_e]['query_budget'], [pre_trained for _ in range((qb_e-qb_s))],
          linestyle='dashed', linewidth=2, label='Pre-trained', color="green")
plt.plot(df.iloc[qb_s:qb_e]['query_budget'], [fine_tuned for _ in range((qb_e-qb_s))],
          linestyle='dashed', linewidth=2, label='Fine-tuned', color="red")
plt.xlabel("Query Budget")
plt.yticks(yticks)
plt.xticks([512, 1024, 2048, 4096])
plt.ylabel("PPL")
plt.legend(loc='upper left', bbox_to_anchor=(0, 0.9))
plt.savefig('plts/query_budget.png')
plt.clf()

plt.plot(df.iloc[p_s:]['p'].astype(str), df.iloc[p_s:]['ppl'],
          linewidth=2, label="PMixED", color='gold', marker='o')
plt.plot(df.iloc[p_s:]['p'].astype(str), [pre_trained for _ in range((p_e-p_s))],
          linestyle='dashed', linewidth=2, label='Pre-trained', color="green")
plt.plot(df.iloc[p_s:]['p'].astype(str), [fine_tuned for _ in range((p_e-p_s))],
          linestyle='dashed', linewidth=2, label='Fine-tuned', color="red")
plt.xlabel("Subsample probability")
plt.yticks(yticks)
#plt.xticks([1/16, 1/8, 1/4, 1/2, 1])
plt.ylabel("PPL")
plt.legend(loc='upper left', bbox_to_anchor=(0, 0.9))
plt.savefig('plts/probability.png')
plt.clf()
