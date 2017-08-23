from sklearn.preprocessing import StandardScaler
from utils import *

#predicting gains/losses for MTL model compared to single task ones using some features

nb_train = 200 #nb text for the training
exp = 2 #exp 1 or 2
ep = 80 #epoch used for previsions
number_test = 0

tasks, list_ind, list_exp = creation(exp)

f = open("exp" + str(exp) + "_" + str(number_test) + "/results" + str(nb_train) + "/efficiency_epoch" + str(ep), "w")
f.write("Epoch " + str(ep) + "\n\n")
 
tableaux = np.load("exp" + str(exp) + "_" + str(number_test) + "/results" + str(nb_train) + "/" + str(ep) + "array_features_log_reg"+ ".npz")
X_base = tableaux['arr_0']
#X_base = charge_params(exp, nb_train, list_exp, number_test, ep)

scaler = StandardScaler().fit(X_base)
X = scaler.transform(X_base)
X_only_curve = np.delete(X, [18, 3, 10, 11, 19, 2, 4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23], 1)
X_non_curve = np.delete(X, [0, 1, 8, 9, 16, 17, 24, 25, 26, 27, 28], 1)
X_non_test = np.delete(X, [25, 26, 24, 29], axis = 1)
X_non_ratio = np.delete(X, [16, 17, 18, 19, 20, 21, 22, 23], axis = 1)

y, nb_true, nb_false = charge_results(exp, nb_train, list_exp,  number_test, ep)
print("Data charged")
f.write("nb true " + str(nb_true) + "\n")
f.write("nb false " + str(nb_false) + "\n\n")

f.write("All features\n")
means_f1 , means_acc = log_reg(X, y)
f.write("Accuracy " + str(means_acc.mean()) + " ")
f.write(str(means_acc.std()) + "\n")

f.write("Without test data\n")
means_f1 , means_acc = log_reg(X_non_test, y)
f.write("Accuracy " + str(means_acc.mean()) + " ")
f.write(str(means_acc.std()) + "\n")


f.write("Only curve data\n")
means_f1 , means_acc = log_reg(X_only_curve, y)
f.write("Accuracy " + str(means_acc.mean()) + " ")
f.write(str(means_acc.std()) + "\n")

f.write("Without curve data\n")
means_f1 , means_acc = log_reg(X_non_curve, y)
f.write("Accuracy " + str(means_acc.mean()) + " ")
f.write(str(means_acc.std()) + "\n")

f.write("Without ratio data\n")
means_f1 , means_acc = log_reg(X_non_ratio, y)
f.write("Accuracy " + str(means_acc.mean()) + " ")
f.write(str(means_acc.std()) + "\n")


#obtain coefficient values for logistic regression
lr = LogisticRegression()
f.write("\n\nCoeffs ")
lr.fit(X, y)
temp = lr.coef_[0]
f.write(str(temp))
f.write("\n\n")
feats = sorted([[i, temp[i]] for i in range(len(X[0]))], key = lambda x: abs(x[1]), reverse = True)
for i in range(30):
    f.write("\n" + str(number_to_feature[feats[i][0]]) + "&" + str(feats[i][1]) + "\\\\")
f.write("\n\n")

#obtain coefficient values for logistic regression
lr = LogisticRegression()
f.write("\n\nCoeffs for non ratio parameters")
lr.fit(X_non_ratio, y)
temp = lr.coef_[0]
f.write(str(temp))
f.write("\n\n")
feats = sorted([[i, temp[i]] for i in range(len(X_non_ratio[0]))], key = lambda x: abs(x[1]), reverse = True)
for i in range(22):
    if feats[i][0] <16: 
        f.write("\n" + str(number_to_feature[feats[i][0]]) + "&" + str(feats[i][1]) + "\\\\")
    if feats[i][0]  > 15: 
        f.write("\n" + str(number_to_feature[feats[i][0]+8]) + "&" + str(feats[i-8][1]) + "\\\\")
f.write("\n\n")


f.close()  
