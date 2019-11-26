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
    #print(data)
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
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import VotingRegressor
    
    # Support Vector Regressor -> find best margin
    best_c = find_best_c(X_1, X_2, Y_1, Y_2)
    svr_reg = SVR(C=best_c, gamma="auto")
    svr_reg.fit(X_1,Y_1)
    
    # Find the best number of hidden layers for the MLP
    best_hidden = find_best_layers(X_1, X_2, Y_1, Y_2)
    mlp_reg = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(best_hidden), random_state = 1)
    mlp_reg.fit(X_1, Y_1)
    
    # Find the best number of neighbors for the KNN Regressor
    best_neighbors = find_best_number_neighbors(X_1, X_2, Y_1, Y_2)
    knn_reg = KNeighborsRegressor(n_neighbors = best_neighbors)
    knn_reg.fit(X_1,Y_1)
    
    # Logistic Regressor
    log_reg = LogisticRegression()
    log_reg.fit(X_1,Y_1)   
    
    
    #ensemble of all previous
    voting_reg = VotingRegressor(estimators=[('knn', knn_reg), ('mlp', mlp_reg), ('svr', svr_reg)])
    voting_reg.fit(X_1, Y_1)
    
    score_log = log_reg.score(X_2, Y_2)
    score_knn = knn_reg.score(X_2, Y_2)
    score_svr = svr_reg.score(X_2, Y_2)
    score_mlp = mlp_reg.score(X_2, Y_2)
    score_ensemble = voting_reg.score(X_2, Y_2)
    
    
                
    #score_ensemble =
    print("\n\n===========================================================")
    print ("Logestic Regression accuracy: {0}".format(score_log.mean()))
    print ("K-Nearest Kneighbors accuracy: {0}".format(score_knn.mean()))
    print ("Support Vector Regression accuracy: {0}".format(score_svr.mean()))
    print ("Multi Layer Perceptron Regression accuracy: {0}".format(score_mlp.mean()))
    print ("Ensemble of all 4 accuracy: {0}".format(score_ensemble.mean()))
    print("===========================================================\n\n")
    
    print_matrix(knn_reg, X_2, Y_2, 'KNN')
    print_matrix(log_reg, X_2, Y_2, 'Log')
    #print_matrix(svr_reg, X_2, Y_2, 'SVR')
    #print_matrix(mlp_reg, X_2, Y_2, 'MLP')
    #print_matrix(voting_reg, X_2, Y_2, 'Ensemble')
    
    #scores = [log_reg.score(X_2, Y_2), knn_reg.score(X_2, Y_2), svr_reg.score(X_2, Y_2), mlp_reg.score(X_2, Y_2), voting_reg.score(X_2, Y_2)]
    #models = [log_reg, knn_reg, svr_reg, mlp_reg, voting_reg]
    return svr_reg
###############################################################################
#
# Finds the best Single Layer Size for a Multi Layer Perceptron Regressor
#
def find_best_layers(X_1, X_2, Y_1, Y_2):
    from sklearn.neural_network import MLPRegressor
    print("Finding the best layer size for the MLP Regression")
    layer_size = 13
    best_score = 0
    best_hidden = layer_size
    TOTAL_TEST_SIZE = 20
    while (layer_size < TOTAL_TEST_SIZE):
        nn_clf = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(layer_size), random_state = 1)
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
# Finds the best Margin for the Support Vector Regressor by performing grid search
#
def find_best_c(X_1, X_2, Y_1, Y_2):
    from sklearn.svm import SVR
    print("Finding best Margin (C) for Support Vector Regression")
    c = 0.0001
    best_c = c
    best_score = 0
    TOTAL_TEST_SIZE = 150
    while (c < TOTAL_TEST_SIZE):
        svr_reg = SVR(C=c, gamma="auto")
        svr_reg.fit(X_1,Y_1)
        
        score_svr = svr_reg.score(X_2, Y_2)
        print("Testing Margin Value: " + str(c), "/", str(TOTAL_TEST_SIZE), "\tFound score:", str(score_svr))
        if (score_svr > best_score):
            best_score = score_svr
            best_c = c
        c += 10
        
    print("best c: " + str(best_c))
    return best_c

###############################################################################
#
# Finds the best number of neighbors for a K nearest neighbors Regressor
#
def find_best_number_neighbors(X_1, X_2, Y_1, Y_2):
    from sklearn.neighbors import KNeighborsRegressor
    print("Finding Best Nearest Neighbors")
    best_num_neighbors = 1
    best_score = 0
    i = 1
    TOTAL_TEST_SIZE = 10
    while (i < TOTAL_TEST_SIZE):
        
        knn_reg = KNeighborsRegressor(n_neighbors= i)
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
    print("Matrix for mode: " + model_name)
    output = model.predict(X_2)
    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(output, Y_2)
    print (matrix)
    return
    
###############################################################################
#
# Dumps the regressors into a file
#
def dump_regs(best_model):
    from sklearn.externals import joblib
    joblib.dump(best_model, 'model/best_model_nb.pkl')
    return

###############################################################################
main()