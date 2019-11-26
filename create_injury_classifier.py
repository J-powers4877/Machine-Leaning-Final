#
# Injury Prediction
#

###############################################################################
def main():
    data = load_data()
    X_1, X_2, Y_1, Y_2, columns = split_data(data)
    best_model = train_clf(X_1, X_2, Y_1, Y_2, columns)
    dump_regs(best_model)
    return

###############################################################################
#
# Load the Dataset from a CSV file
#
def load_data():
    import pandas as pd
    path='injured-and-uninjured.csv'
    df=pd.read_csv(path, sep=',', header=0)
    data = df.drop(df.columns[0], axis=1)
    data = data.to_dict(orient='records')
    return data

###############################################################################
#
# Split the imported CSV file into Training and Testing Datasets
# 
def split_data(data):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.model_selection import train_test_split
    from pandas import DataFrame
    vec = DictVectorizer()

    df_data = vec.fit_transform(data).toarray()
    feature_names = vec.get_feature_names()
    df_data = DataFrame(
    df_data,
    columns=feature_names)
    #print(feature_names)
    outcome_feature = df_data['Injured']
    target_features = df_data.drop('Injured', axis=1)
    
    """
    X_1: independent variables for first data set
    Y_1: dependent (target) variable for first data set
    X_2: independent variables for the second data set
    Y_2: dependent (target) variable for the second data set
    """
    
    X_1, X_2, Y_1, Y_2 = train_test_split(
            target_features, outcome_feature, test_size=0.5, random_state=0)
    
    return X_1, X_2, Y_1, Y_2, feature_names

###############################################################################
#
# Trains Each of the classifiers and finds prints their accuracy scores
#
def train_clf(X_1, X_2, Y_1, Y_2, columns):                       
    #from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import svm
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import VotingClassifier
    
    # Support Vector Classifior -> find best margin
    best_c = find_best_c(X_1, X_2, Y_1, Y_2)
    svm_clf = svm.SVC(C=best_c, gamma="auto")
    svm_clf.fit(X_1,Y_1)
    
    # Find the best number of hidden layers for the MLP
    best_hidden = find_best_layers(X_1, X_2, Y_1, Y_2)
    mlp_clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(best_hidden), random_state = 1)
    mlp_clf.fit(X_1, Y_1)
    
    # Find the best number of neighbors for the KNN Classifier
    best_neighbors = find_best_number_neighbors(X_1, X_2, Y_1, Y_2)
    knn_clf = KNeighborsClassifier(n_neighbors = best_neighbors)
    knn_clf.fit(X_1,Y_1)
    
    # Logistic Regression
    #log_reg = LogisticRegression()
    #log_reg.fit(X_1,Y_1)   
    
    
    #ensemble of all previous
    voting_clf = VotingClassifier(estimators=[('knn', knn_clf), ('mlp', mlp_clf), ('svr', svm_clf)])
    voting_clf.fit(X_1, Y_1)
    
    #score_log = log_reg.score(X_2, Y_2)
    score_knn = knn_clf.score(X_2, Y_2)
    score_svr = svm_clf.score(X_2, Y_2)
    score_mlp = mlp_clf.score(X_2, Y_2)
    score_ensemble = voting_clf.score(X_2, Y_2)
    
    
                
    #score_ensemble =
    print("\n\n===========================================================")
    #print ("Logestic Classification accuracy: {0}".format(score_log.mean()))
    print ("K-Nearest Kneighbors accuracy: {0}".format(score_knn.mean()))
    print ("Support Vector Classification accuracy: {0}".format(score_svr.mean()))
    print ("Multi Layer Perceptron Classification accuracy: {0}".format(score_mlp.mean()))
    print ("Ensemble of the three Classifiers KNN, SVM, MLP accuracy: {0}".format(score_ensemble.mean()))
    print("===========================================================\n\n")
    
    print_matrix(knn_clf, X_2, Y_2, 'KNN')
    #print_matrix(log_clf, X_2, Y_2, 'Log')
    print_matrix(svm_clf, X_2, Y_2, 'SVM')
    print_matrix(mlp_clf, X_2, Y_2, 'MLP')
    print_matrix(voting_clf, X_2, Y_2, 'Ensemble')
    
    #scores = [log_reg.score(X_2, Y_2), knn_reg.score(X_2, Y_2), svr_reg.score(X_2, Y_2), mlp_reg.score(X_2, Y_2), voting_reg.score(X_2, Y_2)]
    #models = [log_reg, knn_reg, svr_reg, mlp_reg, voting_reg]
    return svm_clf
###############################################################################
#
# Finds the best Single Layer Size for a Multi Layer Perceptron Classifior
#
def find_best_layers(X_1, X_2, Y_1, Y_2):
    from sklearn.neural_network import MLPClassifier
    print("Finding the best layer size for the MLP Classification")
    layer_size = 13
    best_score = 0
    best_hidden = layer_size
    TOTAL_TEST_SIZE = 20
    while (layer_size < TOTAL_TEST_SIZE):
        nn_clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(layer_size), random_state = 1)
        nn_clf.fit(X_1, Y_1)   
        score = nn_clf.score(X_2, Y_2, sample_weight=None)
        print("Testing Layer size: " + str(layer_size), "/", str(TOTAL_TEST_SIZE), "\tFound score:", str(score))
        
        if (score > best_score and score != 1.0):
            best_score = score
            best_hidden = layer_size
            
        layer_size += 1
    print("Best Hidden Layer Size: " + str(best_hidden))
    return best_hidden

###############################################################################
#
# Finds the best Margin for the Support Vector Classifior by performing grid search
#
def find_best_c(X_1, X_2, Y_1, Y_2):
    from sklearn import svm
    print("Finding best Margin (C) for Support Vector Machine Classification")
    c = 0.1
    best_c = c
    best_score = 0
    TOTAL_TEST_SIZE = 4
    while (c < TOTAL_TEST_SIZE):
        svm_reg = svm.SVC(C=c, gamma="auto")
        svm_reg.fit(X_1,Y_1)
        
        score_svm = svm_reg.score(X_2, Y_2)
        print("Testing Margin Value: " + str(c), "/", str(TOTAL_TEST_SIZE), "\tFound score:", str(score_svm))
        if (score_svm > best_score):
            best_score = score_svm
            best_c = c
        c += 0.5
        
    print("best c: " + str(best_c))
    return best_c

###############################################################################
#
# Finds the best number of neighbors for a K nearest neighbors Classifior
#
def find_best_number_neighbors(X_1, X_2, Y_1, Y_2):
    from sklearn.neighbors import KNeighborsClassifier
    print("Finding Best Nearest Neighbors")
    best_num_neighbors = 1
    best_score = 0
    i = 1
    TOTAL_TEST_SIZE = 10
    while (i < TOTAL_TEST_SIZE):
        
        knn_reg = KNeighborsClassifier(n_neighbors= i)
        knn_reg.fit(X_1,Y_1)
        score_knn = knn_reg.score(X_2, Y_2)
        print("Testing Neighbors: " + str(i), "/", str(TOTAL_TEST_SIZE), "\tFound score:", str(score_knn))
        if (score_knn > best_score):
            best_score = score_knn
            best_num_neighbors = i
        i += 1
    
    print("Found best neighbors: " + str(best_num_neighbors))
    return best_num_neighbors

###############################################################################
#
# Prints the confusion Matrix for a given model
#
def print_matrix(model, X_2, Y_2, model_name):
    print("\n========================================")
    print("Matrix for model: " + model_name)
    print("Injured  Uinjured")
    output = model.predict(X_2)
    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(output, Y_2)
    print (matrix)
    print("========================================\n")
    return
    
###############################################################################
#
# Dumps the Classifiors into a file
#
def dump_regs(best_model):
    FILE_DUMP = 'model/model_nb.pkl'
    from sklearn.externals import joblib
    joblib.dump(best_model, FILE_DUMP)
    print("Dumped model to: " + FILE_DUMP)
    return

###############################################################################
main()