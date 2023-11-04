import matplotlib.pyplot as plt 
import csv
import pandas as pd

file_name = "results.csv"

df = pd.read_csv(file_name)

eps_s, eps_e = 0, 5 
ens_s, ens_e = 5, 9 
alp_s, alp_e = 9, 14 
qb_s, qb_e = 14, 18 
p_s, p_e = 18, 22

pre_trained = 40.39
fine_tuned = 25.96
yticks = [25, 30, 35, 40]

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
plt.legend()
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
plt.xticks([3, 4, 5, 6, 7])
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
plt.legend()
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
plt.legend()
plt.savefig('plts/probability.png')
plt.clf()

data = {"Pre-trained": pre_trained, "Fine-Tuned": fine_tuned, "DP-SGD": 34.33, "PMixED": 33.30}

bars = plt.bar(data.keys(), data.values(), width=0.4)
bars[0].set_color('green')
bars[1].set_color('red')
bars[2].set_color('blue')
bars[3].set_color('gold')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x()+0.2, yval+0.1, yval, ha='center')
#plt.text(35.98, 35.98, 35.98, ha="center")

plt.ylabel("PPL")
plt.savefig("plts/comparison.png")
plt.clf()