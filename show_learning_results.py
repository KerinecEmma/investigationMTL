import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils import *

#Useful to plot graphics and write main data characteristics (such as mean gain, number of improvment...) in 'gain' files

nb_train = 200 #size of the training set
iters = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95] #epochs for which we want datas
nb_epochs = 100 #number max of epochs for plotting curves
step = 1 #precision of the curve


for exp in range(1, 3):

	list_tasks, list_ind, list_exp = creation(exp)

	f = open("gains_" + str(exp), "w")

	for it in iters:
		f.write(str(it) + "\n")
		if exp == 1:
			#for a list of epochs print the relative gains and loss for f1 score using MTL compare to single task learning
			#not possible for exp 2 (too much datas)
			"""for cat in range(3):
				mean_result = []
				for testi in range(5):
				
					fig, ax = plt.subplots()
					data = np.full((len(list_ind[cat]), len(list_ind[cat])), 600.)
					mean=[]
					for i in range(len(list_ind[cat])):
						task = list_ind[cat][i]
						auxs = make_aux(task, list_exp)

						resultsimple = get_f1S("exp1_" + str(testi) +"/results200/simple/"+ str(task[0])+"_"+str(task[1]),  it)
						_,_,a0 = num_class(exp, task[0])
						_,_,a1 = num_class(exp, task[1])
			
						multis = []
						#f.write("Result MLT: ")
						for j in range(len(list_ind[cat])):
							if list_ind[cat][j] in auxs:
								resultmulti = get_f1M("exp1_" + str(testi) +"/results200/multi/"+ str(task[0])+"_"+str(task[1])+"_"+str(list_ind[cat][j][0])+"_"+str(list_ind[cat][j][1]), it)
								multis += [resultmulti]
								#f.write(str(resultmulti) + "  ")
								if resultsimple == 0:
									data[i][j] = float('nan')
								else:
									data[i][j] = (resultmulti - resultsimple) / (resultsimple)
								plt.text(i + 0.5, j + 0.5, '%.2f' % round(data[i, j], 3),horizontalalignment = 'center',verticalalignment = 'center',)
						if resultsimple == 0:
							mean += [0]
						else:
							mean += [(resultmulti - resultsimple) / (resultsimple)]
					mean_result += [data]
					print(len(mean_result))
					print(len(mean_result[0]))
					print(mean_result[0])
					print(np.mean(mean_result, axis=0))

				masked_array = np.ma.array(np.mean(mean_result, axis=0), mask = (data == 600))
				heatmap = plt.pcolor(masked_array )
				plt.colorbar(heatmap)

				ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor = False)
				ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor = False)
				noms = [["m-h", "m-b", "a_h", "a_b"],["g-i", "g-m", "wm-i", "wm-m", "wx-i", "wx-m"],["m-m","m-a","m-C", "g-m","g-a","g-C","M-m","M-a","M-C"]]
				ax.set_xticklabels(noms[cat])
				ax.set_yticklabels(noms[cat])
				plt.show()"""
	
		#obtain main characteristics
		if exp == 1 :
			mean_gains = []
			mean_f1 = []
			mean_F1 = []
			mean_impr = []
			for testi in range(5):
				gains = np.zeros(len(list_exp))
				impr = 0 
				f1s = []
				F1s = []
				for i in range(len(list_exp)):
					f1 = get_f1S(name(exp, nb_train, list_exp[i][0], testi, False), it)
					f1s += [f1]
					F1 = get_f1M(name(exp, nb_train, list_exp[i], testi, False) + "split", it)
					F1s += [F1]
					if F1 > f1:
						impr += 1
					if f1 != 0:
						gains[i] = (F1 - f1) / (f1)
						#f.write(str(gains[i]) + " ")
					mean_gains += [gains]
					mean_f1 += f1s
					mean_F1 += F1s
				mean_impr += [impr]
			f.write("\nGlobal mean: ")
			f.write(str(np.mean(np.mean(mean_gains, axis = 0))) + "\n\n")
			f.write("nb improvment: " + str(np.mean(mean_impr, axis = 0)) + " compare to " + str(len(list_exp)) + "\n\n")
			f.write("mean simple " + str(np.mean(np.mean(mean_f1, 0))) + "\n")
			f.write("mean multi " + str(np.mean(np.mean(mean_F1, 0))) + "\n\n\n")

		else:
			gains = np.zeros(len(list_exp))
			impr = 0 
			f1s = []
			F1s = []
			for i in range(len(list_exp)):
				f1 = get_f1S(name(exp, nb_train, list_exp[i][0], 0, False),  it)
				f1s += [f1]
				F1 = get_f1M(name(exp, nb_train, list_exp[i], 0, False) + "split", it)
				F1s += [F1]
				if F1 > f1:
					impr += 1
				if f1 != 0:
					gains[i] = (F1 - f1) / (f1)
					#f.write(str(gains[i]) + " ")
			f.write("\nGlobal mean: ")
			f.write(str(np.mean(gains)) + "\n\n")
			f.write("nb improvment: " + str(impr) + " compare to " + str(len(list_exp)) + "\n\n")
			f.write("mean simple " + str(np.mean(f1s)) + "\n")
			f.write("mean multi " + str(np.mean(F1s)) + "\n\n\n")

	f.close()

	#Curve of progression of f1 score depending of the number of epochs for single and multi task learning
	if exp == 1:
		mean_score_s = []
		mean_score_m = []
		for testi in range(5):
			score_s = np.zeros((len(list_tasks), nb_epochs / step))
			score_m = np.zeros((len(list_exp), nb_epochs / step))
			for j in range(nb_epochs / step):
				for i, task in enumerate(list_tasks):	
					score_s[i][j] = get_f1S(name(exp, nb_train, task, 0, False),  j * step)
				for i, ex in enumerate(list_exp):
					score_m[i][j] = get_f1M(name(exp, nb_train, ex, 0, False) + "split", j * step)

			mean_score_s += [score_s]
			mean_score_m += [score_m]


		evol_simple = np.mean(np.mean(mean_score_s, 0), 0)
		evol_multi = np.mean(np.mean(mean_score_m, 0), 0)

		#Show graphics
		plt.plot([i * step for i in range(nb_epochs / step)], evol_simple, '+')
		plt.plot([i * step for i in range(nb_epochs / step)], evol_multi, '.')
		plt.show()

		plt.plot([i * step for i in range(nb_epochs / step)], evol_simple)
		plt.show()
		plt.plot([i * step for i in range(nb_epochs / step)], evol_multi)
		plt.show()

	else:
		score_s = np.zeros((len(list_tasks), nb_epochs / step))
		score_m = np.zeros((len(list_exp), nb_epochs / step))
		for j in range(nb_epochs / step):
			for i, task in enumerate(list_tasks):	
				score_s[i][j] = get_f1S(name(exp, nb_train, task, 0, False),  j * step)
			for i, ex in enumerate(list_exp):
				score_m[i][j] = get_f1M(name(exp, nb_train, ex, 0, False) + "split", j * step)
		evol_simple = np.mean(score_s, 0)
		evol_multi = np.mean(score_m, 0)

		#Show graphics
		plt.plot([i * step for i in range(nb_epochs / step)], evol_simple, '+')
		plt.plot([i * step for i in range(nb_epochs / step)], evol_multi, '.')
		plt.show()

		plt.plot([i * step for i in range(nb_epochs / step)], evol_simple)
		plt.show()
		plt.plot([i * step for i in range(nb_epochs / step)], evol_multi)
		plt.show()

