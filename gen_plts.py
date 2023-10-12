import matplotlib.pyplot as plt 
import csv
import pandas as pd

file_name = "results.csv"

df = pd.read_csv(file_name)

plt.plot(df.iloc[:5]['epsilon'], df.iloc[:5]['ppl'], linewidth=2, label="PMixED", color='gold', marker='o')
plt.plot(df.iloc[:5]['epsilon'], [36.74 for _ in range(5)], linestyle='dashed', linewidth=2, label='Public', color="green")
plt.plot(df.iloc[:5]['epsilon'], [22.83 for _ in range(5)], linestyle='dashed', linewidth=2, label='Private', color="red")
plt.xlabel("Epsilon")
plt.yticks([25, 30, 35])
plt.xticks([2, 4, 6, 8, 10])
plt.ylabel("PPL")
plt.legend(loc='center left')
plt.savefig('plts/epsilon.png')
plt.clf()

plt.plot(df.iloc[5:9]['num_ensemble'], df.iloc[5:9]['ppl'], linewidth=2, label="PMixED", color='gold', marker='o')
plt.plot(df.iloc[5:9]['num_ensemble'], [36.74 for _ in range(4)], linestyle='dashed', linewidth=2, label='Public', color="green")
plt.plot(df.iloc[5:9]['num_ensemble'], [22.83 for _ in range(4)], linestyle='dashed', linewidth=2, label='Private', color="red")
plt.xlabel("Number of Ensembles")
plt.yticks([25, 30, 35])
plt.xticks([8, 16, 32, 64])
plt.ylabel("PPL")
plt.legend()
plt.savefig('plts/ensemble.png')
plt.clf()

plt.plot(df.iloc[9:14]['alpha'], df.iloc[9:14]['ppl'], linewidth=2, label="PMixED", color='gold', marker='o')
plt.plot(df.iloc[9:14]['alpha'], [36.74 for _ in range(5)], linestyle='dashed', linewidth=2, label='Public', color="green")
plt.plot(df.iloc[9:14]['alpha'], [22.83 for _ in range(5)], linestyle='dashed', linewidth=2, label='Private', color="red")
plt.xlabel("Alpha")
plt.yticks([25, 30, 35])
plt.xticks([3, 4, 5, 6, 7])
plt.ylabel("PPL")
plt.legend()
plt.savefig('plts/alpha.png')
plt.clf()

plt.plot(df.iloc[14:18]['query_budget'], df.iloc[14:18]['ppl'], linewidth=2, label="PMixED", color='gold', marker='o')
plt.plot(df.iloc[14:18]['query_budget'], [36.74 for _ in range(4)], linestyle='dashed', linewidth=2, label='Public', color="green")
plt.plot(df.iloc[14:18]['query_budget'], [22.83 for _ in range(4)], linestyle='dashed', linewidth=2, label='Private', color="red")
plt.xlabel("Query Budget")
plt.yticks([25, 30, 35])
plt.xticks([512, 1024, 2048, 4096])
plt.ylabel("PPL")
plt.legend()
plt.savefig('plts/query_budget.png')
plt.clf()

plt.plot(df.iloc[18:]['p'], df.iloc[18:]['ppl'], linewidth=2, label="PMixED", color='gold', marker='o')
plt.plot(df.iloc[18:]['p'], [36.74 for _ in range(4)], linestyle='dashed', linewidth=2, label='Public', color="green")
plt.plot(df.iloc[18:]['p'], [22.83 for _ in range(4)], linestyle='dashed', linewidth=2, label='Private', color="red")
plt.xlabel("Subsample probability")
plt.yticks([25, 30, 35])
plt.xticks([1/16, 1/8, 1/4, 1/2])
plt.ylabel("PPL")
plt.legend()
plt.savefig('plts/probability.png')
plt.clf()

data = {"Public": 35.98, "Private": 22.61, "DP-SGD": 37.33, "PMixED": 30.80}

bars = plt.bar(data.keys(), data.values(), width=0.4)
bars[0].set_color('green')
bars[1].set_color('red')
bars[2].set_color('blue')
bars[2].set_color('gold')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x()+0.2, yval+0.1, yval, ha='center')
#plt.text(35.98, 35.98, 35.98, ha="center")

plt.ylabel("PPL")
plt.savefig("plts/comparison.png")
plt.clf()