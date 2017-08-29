import os
import sys

sys.path.append(os.getcwd() + "//..")
from graph_features import initGraph
from learning.TagsLoader import TagsLoader
from graphs_mains.features_calculator import featuresCalculator
import graphs_mains.featuresList as featuresList
import pickle
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from learning import myTrain
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import random


currentDirectory = str(os.getcwd())


def init_vertex_dict_by_edges(gnx,trains,tags):
    num_classes = max(tags.values())
    vertex_dict = {}
    count = 0
    nodes = gnx.nodes()
    for node in nodes:
        vertex_dict[node] = {'out': [0 for i in range(int(num_classes)+1)], 'in': [0 for i in range(int(num_classes)+1)]}
    edges = gnx.edges()
    for edge in edges:
        count +=1
        # print (count,' out of: ',len(edges))
        if edge[1] in trains:
            color = int(tags[edge[1]])
            vertex_dict[edge[0]]['out'][color] +=1
        if edge[0] in trains:
            color = int(tags[edge[0]])
            vertex_dict[edge[1]]['in'][color] +=1
    return vertex_dict

def split_train_test(nodes, tags_dict, test_size, random_state):
    tags = []
    for node in nodes:
        tags.append(tags_dict[node])
    if random_state != None:
        x_train, x_test, y_train, y_test = train_test_split(nodes, tags,
                                                        test_size=test_size, random_state=random_state)
    else:
         x_train, x_test, y_train, y_test = train_test_split(nodes, tags,
                                                        test_size=test_size)
    return x_train, x_test, y_train, y_test

def calc_global_features(analysis_type):
    calculator = featuresCalculator()
    features_list = []
    if 'mds' in analysis_type:
        features_list.append('mds')
    if 'features' in analysis_type:
        features_list.extend(featuresList.featuresList(directed=True, analysisType='nodes').getFeatures())
    output_dir = str(currentDirectory) + r'/../data/directed/' + graph_name + '/features'
    result = calculator.calculateFeatures(features_list, file_in, output_dir, directed=True,
                                          analysisType='nodes', parallel=False)
    return result

def calc_features(tests,trains,tags,global_features):
    train_features = []
    test_features = []
    for node in trains:
        node_to_sign = [node]
        [node_to_sign.append(vertex_dict[node]['in'][i]) for i in range(int(max(tags.values()))+1)]
        [node_to_sign.append(vertex_dict[node]['out'][i]) for i in range(int(max(tags.values()))+1)]
        train_features.append(node_to_sign)
        for feature in global_features:
            train_features[-1].extend(global_features[feature][node])
        train_features[-1].append(tags[node])
    for node in tests:
        node_to_sign = [node]
        [node_to_sign.append(vertex_dict[node]['in'][i]) for i in range(int(max(tags.values()))+1)]
        [node_to_sign.append(vertex_dict[node]['out'][i]) for i in range(int(max(tags.values()))+1)]
        test_features.append(node_to_sign)
        for feature in global_features:
            test_features[-1].extend(global_features[feature][node])
        test_features[-1].append(tags[node])
    return train_features, test_features

def write_features_to_file(train_features, test_features):
    with open(output_result_dir + 'train_features.dump', 'wb') as f:
        pickle.dump(train_features, f)
    with open(output_result_dir + 'test_features.dump', 'wb') as f:
        pickle.dump(test_features, f)

def read_features_from_file():
    with open(output_result_dir + 'train_features.dump', 'rb') as f:
        train_features = pickle.load(f)
    with open(output_result_dir + 'test_features.dump', 'rb') as f:
        test_features = pickle.load(f)
    return train_features,test_features

def z_scoring(matrix):
    new_matrix = np.matrix(matrix)

    minimum = np.asarray(new_matrix.min(0))
    for i in range(minimum.shape[1]):
        if minimum[0,i] > 0:
            new_matrix[:,i] = np.log10(new_matrix[:,i])
        elif minimum[0,i] == 0:
            new_matrix[:, i] = np.log10(new_matrix[:, i]+0.1)
        if new_matrix[:,i].std() > 0:
            new_matrix[:,i] = (new_matrix[:,i]-new_matrix[:,i].min())/new_matrix[:,i].std()
    return new_matrix

def get_features(x_test, x_train, gnx, train_features, test_features, tags, type):
    num_of_classes = int(max(tags.values()))
    if type == 'samples':
        train = samples(gnx, x_train, 7)
        test = samples(gnx, x_test, 7)
    if type == 'local':
        train = [x[1:(num_of_classes+1)*2+1] for x in train_features]
        test = [x[1:(num_of_classes+1)*2+1] for x in test_features]
    elif type == 'global':
        train = [x[(num_of_classes+1)*2+1:-1] for x in train_features]
        test = [x[(num_of_classes+1)*2+1:-1] for x in test_features]
    elif type == 'both':
        train = [x[1:-1] for x in train_features]
        test = [x[1:-1] for x in test_features]
    train_tags = [x[-1] for x in train_features]
    test_tags = [x[-1] for x in test_features]
    if type == 'classes':
        train = [[0 for i in range(int(max(list(train_tags))) + 2)] for j in range(len(train_features))]
        test = [[0 for i in range(int(max(list(test_tags))) + 2)] for j in range(len(test_features))]
        train = np.matrix(train)
        test = np.matrix(test)
        for index in range(len(list(train_features))):
            train[index,int(train_tags[index])] = 1
            train[index,-1] = 1
        for index in range(len(list(test_features))):
            test[index,-1] = 1
    else:
        train = z_scoring(train)
        test = z_scoring(test)

    return train, train_tags, test, test_tags

def evaluate_acc(clf, test, test_tags, train, train_tags,f_output):
    predictions_proba = clf.predict_proba(test)
    predictions = np.argmax(predictions_proba,axis=1)
    accuracy_test = accuracy_score(test_tags, predictions)
    print ('Accuracy Test:', accuracy_test)
    f_output.writelines('Accuracy Train:' + str(accuracy_test) + '\n')
    predictions_proba = clf.predict_proba(train)
    predictions = np.argmax(predictions_proba,axis=1)
    accuracy_train = accuracy_score(train_tags, predictions)
    print ('Accuracy Train:',accuracy_train)
    f_output.writelines('Accuracy Train:'+str(accuracy_train) + '\n')

def evaluate_confusion_metric_test(clf, test, test_tags, train, train_tags,f_output):
        y_pred = clf.predict_proba(test)
        y_pred = [np.argmax(lst) for lst in y_pred]
        y_true = [int(i) for i in test_tags]
        confusion_matrix_result = metrics.confusion_matrix(y_true,y_pred)
        confusion_matrix_result = confusion_matrix_result.astype('float') / confusion_matrix_result.sum(axis=1)[:, np.newaxis]
        plot_confusion_matrix(confusion_matrix_result,range(int(max(test_tags))+1),plot_file_name=f_output)

def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              plot_file_name='confusion matrix.png'):

        print (normalize)
        if (normalize):
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        df_cm = pd.DataFrame(cm, index=[i for i in classes],
                             columns=[i for i in classes])

        plt.figure()
        sn.heatmap(df_cm, annot=True)
        plt.title(title)
        plt.savefig(plot_file_name)

def preform_learning(train, train_tags, test, test_tags, f_output,f_cm, test_size, graph_name,analysis_type, deep = False):
    if not deep:
        # algos = ['adaBoost', 'RF', 'L-SVM', 'RBF-SVM', 'SGD']
        algos = ['L-SVM']#['adaBoost', 'RF', 'L-SVM', 'RBF-SVM']
        for algo in algos:
            print (algo)
            f_output.writelines(algo +'\n')

            if algo == 'adaBoost':
                clf = AdaBoostClassifier(n_estimators=100)
            if algo == 'RF':
                clf = RandomForestClassifier(n_estimators=1000, criterion="gini", min_samples_split=15, oob_score=True,
                                             class_weight='balanced', max_depth=3)
            if algo == 'L-SVM':
                clf = SVC(kernel='linear', class_weight="balanced", C=0.09, probability=True)
            if algo == 'RBF-SVM':
                clf = SVC(class_weight="balanced", C=0.01, probability=True)
            if algo == 'SGD':
                clf = SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1, eta0=0.0,
                                    fit_intercept=True, l1_ratio=0.15, learning_rate='optimal', loss='hinge',
                                    n_iter=5, n_jobs=1, penalty='l2', power_t=0.5, random_state=None, shuffle=True,
                                    verbose=0, warm_start=False)
        # print train
            clf.fit(train, train_tags)
            # if (algo == 'RF'):
            #     print (len(clf.feature_importances_))
            #     print (clf.feature_importances_)
            #     f_output.writelines(str(clf.feature_importances_)+'\n')
            evaluate_acc(clf, test, test_tags, train, train_tags,f_output)
            evaluate_confusion_metric_test(clf,test, test_tags, train, train_tags,f_cm)
    else:
        feature_matrix = np.append(train,test,0)
        train_tags.extend(test_tags)
        x_train.extend(x_test)
        train_size = 1.0-test_size
        myTrain.deepLearning(feature_matrix, train_tags, x_train, gnx, train_size, graph_name, analysis_type)
        # from keras.models import Sequential
        # from keras.layers import Dense, Dropout
        # from keras.regularizers import l2, l1_l2

        # clf = Sequential()
        # clf.add(
        #     Dense(100, activation="relu", kernel_initializer="he_normal", input_dim=train.shape[1]))
        # # self.classifier.add(Dropout(0.5))
        # # self.classifier.add(Dense(100, init='he_normal', activation='relu', W_regularizer=l2(0.5)))
        # clf.add(Dropout(0.5))
        # # clf.add(Dense(1, init='uniform', activation='sigmoid', W_regularizer=l1_l2(0.6)))
        # clf.add(Dense(1, init='uniform', activation='sigmoid'))
        # clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # clf.fit(train, train_tags, validation_data=[test, test_tags], epochs=250,
        #         batch_size=10, verbose=2)




def samples(gnx,nodes,sample_size):
    matrix = []
    for node in nodes:
        ns = gnx.neighbors(node)
        if len(ns) == 0:
            matrix.append([0 for i in range(sample_size*sample_size-sample_size)])
        else:
            if len(ns) > sample_size:
                ns_indexes = random.sample(range(len(ns)), sample_size)
            else:
                 ns_indexes = range(len(ns))
            # the chosen neighbors
            l = []
            for index in ns_indexes:
                l.append(ns[index])
            ns = l
            # the neighbors sub graph
            ns_graph = gnx.subgraph(ns)
            # bubble sort by degree
            for n1 in range(len(ns)):
                for n2 in range(len(ns)-1):
                    l1 = len(ns_graph.neighbors(ns[n2]))
                    l2 = len(ns_graph.neighbors(ns[n2+1]))
                    if l1 > l2:
                        temp = ns[n2]
                        ns[n2] = ns[n2+1]
                        ns[n2+1] = temp
            # the adjacency matrix of the sub graph
            ns_adj = nx.adjacency_matrix(ns_graph, ns)
            ns_adj = ns_adj.todense()
            ns_adj = np.asarray(ns_adj, 'float32')
            # zero padding
            for i in range(sample_size-len(ns)):
                ns_adj = np.append(ns_adj,[[0 for j in range(len(ns))]],0)
            for i in range(sample_size-len(ns)):
                ns_adj = np.append(ns_adj,[[0] for j in range(sample_size)],1)
            ns_mat = []
            for i in range(sample_size):
                for j in range(sample_size):
                    if j!=i:
                        if ns_adj[i,j]!= 0:
                            ns_mat.append(1)
                        else:
                            ns_mat.append(0)
            matrix.append(ns_mat)
    return matrix








# load graph using networkx
# graph_name = 'neighbors_test'
# for graph_name in ['cora', 'citeseer']:
for graph_name in ['cora']:
    file_in = './../data/directed/' + graph_name + '/input/'+graph_name+'.txt'
    print (' start reload graph')
    gnx = initGraph.init_graph(draw=False, file_name=file_in, directed=True, Connected=True)
    print (' finish reload graph')

    # load tags of the graph
    directory_tags_path = str(os.getcwd()) + r'/../data/directed/' + graph_name + '/tags/'
    classification_result = [graph_name + '_tags']
    tagsLoader = TagsLoader(directory_tags_path, classification_result)
    tagsLoader.Load()
    tags_dict = tagsLoader.calssification_to_vertex_to_tag[classification_result[0]]
    output_result_dir = str(os.getcwd()) + r'/../data/directed/' + graph_name + '/features/output/'
    # if graph_name == 'cora':
    #     types = [['samples']]
    # else:
    #     # types = [['mds','global'],['mds','both'],['local'],['features','global'],['features','both'],['mds','features','global'],['mds','features','both']]
    types = [['classes']]
    for analysis_type in types:
        print(str(analysis_type))
        global_features = calc_global_features(analysis_type)[1]
        for test_size in [0.1 * i for i in range(1,10)]:
            # for train_set in range(5):
            print ('test_size',test_size)
            x_train, x_test, y_train, y_test = split_train_test(gnx.nodes(), tags_dict, test_size,random_state=2)
            vertex_dict = init_vertex_dict_by_edges(gnx, x_train, tags_dict)
            train_features, test_features = calc_features(x_test, x_train, tags_dict, global_features)
            # write_features_to_file(train_features, test_features)
            # # train_features, test_features = read_features_from_file()
            train, train_tags, test, test_tags = get_features(x_test, x_train, gnx, train_features, test_features, tags_dict, type=analysis_type[-1])
            f_output = open('./../data/directed/' + graph_name + '/results/'+str(analysis_type)+'.txt', 'w')
            f_cm = './../data/directed/' + graph_name + '/results/'+str(analysis_type)+'_cm.png'
            preform_learning(train,train_tags,test,test_tags,f_output,f_cm,test_size, graph_name,analysis_type, deep=True)



