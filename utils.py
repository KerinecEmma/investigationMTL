import sys
import re
import scipy.sparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from numpy.random import choice
from scipy.optimize import curve_fit
from scipy.stats import entropy
from collections import Counter


#regroupment of categories of texts for problem 1 and 2
problem1 = [[['rec.motorcycles', 'rec.autos'],['rec.sport.baseball', 'rec.sport.hockey']], [['comp.graphics', 'comp.os.ms-windows.misc', 'comp.windows.x'], ['comp.sys.ibm.pc.hardware','comp.sys.mac.hardware']], [['talk.politics.misc','talk.politics.guns', 'talk.politics.mideast'], ['talk.religion.misc', 'alt.atheism', 'soc.religion.christian']]]
problem2 = [['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'], ['rec.motorcycles', 'rec.autos'],['rec.sport.baseball', 'rec.sport.hockey'], ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.windows.x'], ['comp.sys.ibm.pc.hardware','comp.sys.mac.hardware'], ['talk.politics.misc','talk.politics.guns', 'talk.politics.mideast'], ['talk.religion.misc', 'alt.atheism', 'soc.religion.christian']]


def name(exp, nb_train, inp, number_test, array): #for a given exp, nb train and task return the name use for the files
	if array == False:
		if len(inp[0]) != 2:
			return("exp" + str(exp) + "_" + str(number_test) + "/results" + str(nb_train) + "/simple/" + str(inp[0]) + "_" + str(inp[1]))
		else:
			return("exp" + str(exp) + "_" + str(number_test) +  "/results" + str(nb_train) + "/multi/" + str(inp[0][0]) + "_" + str(inp[0][1]) + "_" + str(inp[1][0]) + "_" + str(inp[1][1]))
	else:
		return("exp" + str(exp) + "_" + str(number_test) + "/results" + str(nb_train) + "/array_" + str(inp[0]) + "_" + str(inp[1]))

def num_class(exp, cat): #each category of text is part of some groups and subgroups (usefull for making aux) retunr them
	if exp == 1:
		for i in range(len(problem1)):
			for j in range(len(problem1[i])):
				for k in range(len(problem1[i][j])):
					if problem1[i][j][k] == cat:
						return(i, j, k)
		print("pb in num class")
		print(cat)
		return(-1)
	elif exp == 2:
		for i in range(len(problem2)):
			for j in range(len(problem2[i])):
				if problem2[i][j] == cat:
					return(i, j)
		print("pb in num class")
		print(cat)
	return(-1)


def creation(exp): #for a given experience return the list of tasks and the list of experiences (= couple main task - auxiliary task)
	if exp == 1 : 
		def make_tasks():
			list_tasks = []
			list_ind = []
			for k in range(len(problem1)):
				temp=[]
				for i in range(len(problem1[k])):
					for j in range(i + 1, len(problem1[k])):
						for i1 in range(len(problem1[k][i])):
							for j1 in range(len(problem1[k][j])):
								list_tasks += [[problem1[k][i][i1], problem1[k][j][j1]]]
								temp += [[problem1[k][i][i1], problem1[k][j][j1]]]
				list_ind += [temp]
			return(list_tasks, list_ind)

		def find_aux(target):
			cat00, cat01, cat02 = num_class(exp, target[0])
			cat10, cat11, cat12 = num_class(exp, target[1])
			if cat10 != cat00 or cat01 == cat11:
				print("problem in find aux")
				print(target)
			result = []
			for i in range(len(problem1[cat00][cat01])):
				if i != cat02: 
					for j in range(len(problem1[cat00][cat11])):
						if j != cat12:
							result += [[target, [problem1[cat00][cat01][i], problem1[cat00][cat11][j]]]]
			return(result)

		def find_list(tasks): 
			liste = []
			for i in tasks:
				liste += find_aux(i)
			new_list = []
			for i in range(len(liste)):
				ok = True
				for j in range(i+1, len(liste)):
					if liste[i] == liste[j]:
						ok=False
				if ok:
					new_list += [liste[i]]
			return(new_list)

		list_tasks, list_ind  = make_tasks()
		list_exp = find_list(list_tasks)
		return(list_tasks, list_ind, list_exp)


	elif exp == 2:
		def make_tasks():
			list_tasks = []
			list_ind = []
			temp=[]
			for i in range(len(problem2)):
				for j in range(i + 1, len(problem2)):
					for i1 in range(len(problem2[i])):
						for j1 in range(len(problem2[j])):
							list_tasks += [[problem2[i][i1], problem2[j][j1]]]
							temp += [[problem2[i][i1], problem2[j][j1]]]
			list_ind += [temp]
			return(list_tasks, list_ind)

		def find_aux(target):
			cat00, cat01 = num_class(exp, target[0])
			cat10, cat11 = num_class(exp, target[1])
			if cat00 == cat10:
				print("problem in find aux")
				print(target)
			result = []
			for i in range(len(problem2[cat00])):
				if i != cat01: 
					for j in range(len(problem2[cat10])):
						if j != cat11:
							result += [[target, [problem2[cat00][i], problem2[cat10][j]]]]
			return(result)

		def find_list(tasks): 
			liste = []
			for i in tasks:
				liste += find_aux(i)
			new_list = []
			for i in range(len(liste)):
				ok = True
				for j in range(i+1, len(liste)):
					if liste[i] == liste[j]:
						ok=False
				if ok:
					new_list += [liste[i]]
			return(new_list)

		list_tasks, list_ind  = make_tasks()
		list_exp = find_list(list_tasks)
		return(list_tasks, list_ind, list_exp)

	else:
		return(-1)

def make_aux(task, list_exp): #for a task return the list of corresponding auxiliary tasks
	aux=[]
	for i in range(len(list_exp)):
		if list_exp[i][0]== task:
			aux+=[list_exp[i][1]]
	return(aux)

def splitt(data, target): #a training set is composed of text form to gategories, split them
	train0 = []
	train1 = []
	for i in range(len(data)):
		if target[i] == 1:
			train1 += [data[i]]
		else:
			train0 += [data[i]]
	return(train0, train1)

def JSD(a, b): #Jensen Shannon Divergence
	def aux(P, Q):
	    _P = P / np.linalg.norm(P, ord = 1)
	    _Q = Q / np.linalg.norm(Q, ord = 1)
	    _M = 0.5 * (_P + _Q)
	    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

	d0=(a[0])
	for i in range(1,len(a)):
		d0+=a[i]
	e0=d0.split()
	c0=Counter(e0)

	d1=(b[0])
	for i in range(1,len(b)):
		d1+=b[i]
	e1=d1.split()
	c1=Counter(e1)

	liste = e0+e1
	liste=list(set(liste))
	dist0 = np.array([c0[i] for i in liste])
	dist1 = np.array([c1[i] for i in liste])

	return(aux(dist0, dist1))


def make_inputs(exp, task, nb_features, nb_train, nb_ex): 	#return training and tests sets (tf-idf form)	
	vectorizer = TfidfVectorizer(max_features = nb_features)
	x_test = fetch_20newsgroups(subset = "test", remove=('headers', 'footers', 'quotes'), categories = task)
	x_train = fetch_20newsgroups(subset = "train", remove=('headers', 'footers', 'quotes'), categories = task)
	newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

	nb_docs = min(len(x_test.data), len(x_train.data))

	if nb_ex > nb_docs:
		print("Change nb_ex for {}".format(nb_docs))
		nb_ex= nb_docs

	train_ids = choice(nb_docs, int(nb_train), replace = False)

	x_train_data = [x_train.data[i] for i in train_ids]
	x_train_target = [x_train.target[i] for i in train_ids]

	vectorizer.fit_transform(newsgroups_train.data)
	trainX = vectorizer.transform(x_train_data).toarray()

	test_ids = choice(nb_docs, int(nb_ex), replace = False)

	x_test_data = [x_test.data[i] for i in test_ids]
	x_test_target = [x_test.target[i] for i in test_ids]

	testX = vectorizer.transform(x_test_data).toarray()

	np.savez_compressed("exp" + str(exp) + "/results" + str(nb_train) + "/array_" + str(task[0]) + "_" + str(task[1]), np.array(trainX), np.array(x_train_target), np.array(testX), np.array(x_test_target), train_ids, test_ids)

	return(np.array(trainX), np.array(x_train_target), np.array(testX), np.array(x_test_target))


def read_learning_curve(exp, nb_train, task,  number_test, iter_max): 
	iter_loss = []
	file = open(name(exp, nb_train, task, number_test, False), "r")
	z = 0
	for line in file:
		if z < iter_max:
			spl = line.split()
			iter_loss.append((int(spl[1].strip(',')), float(spl[3].strip(','))))
		z += 1
	return(iter_loss)


def func(x, a, b, c, d):
	return(a + b * np.log(np.multiply(x, c) + d))

def get_curve_params(exp, nb_train, task, number_test, it):
	xs, ys = zip(*read_learning_curve(exp, nb_train, task, number_test, it))
	try:
		popt, _ = curve_fit(func, xs, ys)
		return(popt[1], popt[2])
	except RuntimeError:
		print("Error - curve_fit failed")
	return()

def get_curve_gradients(exp, nb_train, task, it, number_test, pos = [0.1, 0.25, 0.5, 0.75]):
    def get_gradient_at(xs, ys, i):
        xbefore = xs[i - 1]
        ybefore = ys[i - 1]
        xafter = xs[i + 1]
        yafter = ys[i + 1]
        return((yafter - ybefore) / (xafter - xbefore))
    xs, ys = zip(*read_learning_curve(exp, nb_train, task, number_test, it))
    gradients = []
    for p in pos:
        gradients.append(get_gradient_at(xs, ys, int(len(xs) * p)))
    return(gradients)


def get_f1S(name, iters):
    #gets accuracy of simgle task at iters iterations
	f = open(name, "r")
	acc = 0
	d = f.readlines()
	if len(d) == 0:
		return(-1)
	plop = d[iters]
	plop = plop.strip().replace(',', '').split()
	acc = float(plop[11])
	"""if acc == 0:
		print("acc = 0")
		print(name)"""
	f.close()
	return(acc)

def get_f1M(name, iters):
    #gets accuracy of simgle task at iters iterations
	f = open(name, "r")
	acc = 0
	d=f.readlines()
	if len(d) == 0:
		return(-1)
	plop = d[iters]
	plop = plop.strip().replace(',', '').split()
	acc = float(plop[21])
	"""if acc == 0:
		print("acc = 0")
		print(iters)
		print(name)"""
	f.close()
	return(acc)


def read_sets(exp, nb_train, task, number_test):
	#return training and test set
	tableaux = np.load(name(exp, nb_train, task,  number_test, True) + ".npz")
	tab1 = tableaux['arr_0'] #trainX train_ids
	tab2 = tableaux['arr_1'] #trainY
	tab3 = tableaux['arr_2'] #testX
	tab4 = tableaux['arr_3'] #testY
	tab5 = tableaux['arr_4'] #train_ids
	tab6 = tableaux['arr_5'] #test_ids
	return(tab1, tab2, tab3, tab4, tab5, tab6)  

def recharge(ids, task, test): #need to have training sets of a past task
	if test == False : 
		x_train = fetch_20newsgroups(subset = "train", remove=('headers', 'footers', 'quotes'), categories = task)
		x_train_data = [x_train.data[i] for i in ids]
		x_train_target = [x_train.target[i] for i in ids]
		return(x_train_data, x_train_target)
	else:
		x_test = fetch_20newsgroups(subset = "test", remove=('headers', 'footers', 'quotes'), categories = task)
		x_test_data = [x_test.data[i] for i in ids]
		x_test_target = [x_test.target[i] for i in ids]
		return(x_test_data, x_test_target)

def charge_params(exp, nb_train, liste, number_test, ep):
	#give parameter for logistic regression
	#and save them since it takes time to compute them and someone may want to reuse them

	parameters = np.zeros((len(liste), 30))
	for i in range(len(liste)):
		task = liste[i][0]
		aux = liste[i][1]

		_, _, _, _, ids_train_task, ids_test_task= read_sets(exp, nb_train, task, number_test)
		_, _, _, _, ids_train_aux, _ = read_sets(exp, nb_train, aux, number_test)
		train_task_X, train_task_y = recharge(ids_train_task, task, False)
		test_task_X, test_task_y = recharge(ids_test_task, task, True)		
		X_task_0, X_task_1 = splitt(train_task_X, train_task_y)
		train_aux_X, train_aux_y = recharge(ids_train_aux, aux, False)
		X_aux_0, X_aux_1 = splitt(train_aux_X, train_aux_y )

		dictio_train = {}
		word_train_main = 0.0
		word_test = 0.0
		count_not_seen = 0.0
		for text in train_task_X:
			plop = re.split("\W+", text)
			for word in plop:
				if dictio_train.has_key(word) == False:
					dictio_train[word] = 1
					word_train_main += 1
		dictio_test = {}
		for text in test_task_X:
			plop = re.split("\W+", text)
			for word in plop:
				if dictio_test.has_key(word) == False:
					word_test += 1
					dictio_test[word] = 1
					if dictio_train.has_key(word) == False:
						dictio_train[word] = 1
						count_not_seen += 1

		dictio_train = {}
		word_train_aux = 0.0
		for text in train_aux_X:
			plop = re.split("\W+", text)
			for word in plop:
				if dictio_train.has_key(word) == False:
					dictio_train[word] = 1
					word_train_aux += 1
		
		#main task features
		parameters[i][0] = JSD(X_task_0, X_task_1)
		parameters[i][1] = word_train_main
		parameters[i][2], parameters[i][3] = get_curve_params(exp, nb_train, task, number_test, 150)
		parameters[i][4], parameters[i][5], parameters[i][6], parameters[i][7] = get_curve_gradients(exp, nb_train, task, ep, number_test, pos = [0.1, 0.25, 0.5, 0.75])
		
		#aux task features
		parameters[i][8] = JSD(X_aux_0, X_aux_1)
		parameters[i][9] = word_train_aux
		parameters[i][10], parameters[i][11] = get_curve_params(exp, nb_train, aux, number_test, 150)
		parameters[i][12], parameters[i][13], parameters[i][14], parameters[i][15] = get_curve_gradients(exp, nb_train, aux, ep, number_test, pos = [0.1, 0.25, 0.5, 0.75])

		#ratios for respective main task features
		for l in range(8):
			if parameters[i][l + 8] != 0:
				parameters[i][l + 16] = parameters[i][l] / parameters[i][l + 8]

		#general features
		parameters[i][24] = word_test
		parameters[i][25] = JSD(test_task_X, X_task_0) 
		parameters[i][26] = JSD(test_task_X, X_task_1)
		parameters[i][27] = JSD(X_task_0, X_aux_0)
		parameters[i][28] = JSD(X_aux_1, X_task_1)
		parameters[i][29] = count_not_seen / word_test

	np.savez_compressed("exp" + str(exp) +"_" + str(number_test) + "/results" + str(nb_train) + "/" + str(ep) + "array_features_log_reg", np.array(parameters))

	return(parameters)

def charge_results(exp, nb_train, liste,  number_test, epoch): #give results gain/not gain for logistic regression
	accs = {}
	diffs = {}
	y = []
	plop0 = 0
	plop1 = 0
	pb = []
	for i in range(len(liste)):
		task = liste[i][0]
		aux = liste[i][1]
		accs[str(task)] = get_f1S(name(exp, nb_train, task,  number_test, False), epoch)
		f1 = get_f1M(name(exp, nb_train, liste[i],  number_test, False) +"split", epoch)
		if f1 != -1 :
			accs[str(task) + '_' + str(aux)] = f1
			if accs[str(task) + '_' + str(aux)] > accs[str(task)] : 
				diffs[str(task) + '_' + str(aux)] = 1
				print(liste[i])
				plop0 += 1
			else:
				diffs[str(task) + '_' + str(aux)] = -1
				plop1 += 1
			y.append(diffs[str(task) +'_'+ str(aux)])
		else:
			print("pb")
			print(liste[i])
			pb += [i]
	print(pb)
	return(y, plop0, plop1)


def restrict(exp, nb_train, list_exp,  number_test, epoch, limit):	#return a list of task with restriction on the gain/losses between multi and single models
	liste_restrict = []
	compt = 0
	for i in range(len(list_exp)):
		f1 = get_f1S(name(exp, nb_train, list_exp[i][0],  number_test, False), epoch)
		F1=get_f1M(name(exp, nb_train, list_exp[i], False),  number_test, epoch)
		if f1 == 0: 
			compt += 1
			print(list_exp[i])
		else :
			if abs((F1 - f1) / f1) < limit:
				liste_restrict += [list_exp[i]]
	print(compt)
	return(liste_restrict)


def turn_int(list):
	result=[]
	for i in range(len(list)):
		if list[i] > 0.5:
			result += [1]
		else:
			result += [0]
	return(result)

number_to_feature={	#features used for investigate the gain using MTL
0 : "JSD training set cat. 0 vs cat. 1 main",
1 : "nb words training set main",
2 : "Curve param a main",
3 : "Curve param c main",
4 : "Curve gradient 10\\% main",
5 : "Curve gradient 25\\% main",
6 : "Curve gradient 50\\% main",
7 : "Curve gradient 75\\% main",
8 : "JSD training set cat. 0 vs cat. 1 aux",
9 : "nb words training set aux",
10 : "Curve param a aux",
11 : "Curve param c aux",
12 : "Curve gradient 10\\% aux",
13 : "Curve gradient 25\\% aux" ,
14 : "Curve gradient 50\\% aux",
15 : "Curve gradient 75\\% aux",
16 : "JSD training set cat. 0 vs cat. 1 ratio",
17 : "nb words training set ratio",
18 : "Curve param a ratio",
19 : "Curve param c ratio", 
20 : "Curve gradient 10\\% ratio",
21 : "Curve gradient 25\\% ratio",
22 : "Curve gradient 50\\% ratio",
23 : "Curve gradient 75\\% ratio",
24 : "nb words test set",
25 : "JSD test vs training set cat. 0",
26 : "JSD test vs training set cat. 1",
27 : "JSD training set cat. 0 main vs cat. 0 aux",
28 : "JSD training set cat. 1 main vs cat. 1 aux",
29 : "OOV rate"
}

def log_reg(X, y): #mean perf for 100 run of 5 fold cross val
    means_f1 = []
    """for _ in range(100):
        lr = LogisticRegression()
        shuffle = KFold(n_splits = 5, shuffle = True, random_state = np.random.randint(100))
        scores = cross_val_score(lr, X, y, cv = shuffle, scoring ='f1_micro')
        means_f1.append(scores.mean())
    means_f1 = np.array(means_f1)"""
    means_acc = []
    for _ in range(100):
        lr = LogisticRegression()
        shuffle = KFold(n_splits = 5, shuffle = True, random_state = np.random.randint(100))
        scores = cross_val_score(lr, X, y, cv = shuffle, scoring = 'accuracy')
        means_acc.append(scores.mean())
    means_acc = np.array(means_acc)
    return(means_f1 , means_acc)