import Orange  # version: Orange3==3.32.0
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

names = ['TS2Vec-Floss', 'TS2Vec', 'T-Loss', 'TS-TCC', 'TST', 'TNC', 'DTW']
avranks = [2.633, 2.833, 3.600, 4.300, 4.166, 4.966, 4.300]
datasets_num = 30
CD = Orange.evaluation.compute_CD(avranks, datasets_num, alpha='0.05', test='nemenyi')
Orange.evaluation.scoring.graph_ranks(avranks, names, cd=CD, width=8, textspace=1.5, reverse=True)
plt.tight_layout()
plt.savefig("classification.pdf", dpi=600, format="pdf")
plt.show()