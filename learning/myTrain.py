import math
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.nan)
import networkx as nx
import warnings
import os


from scipy.sparse import (spdiags, SparseEfficiencyWarning, csc_matrix,
    csr_matrix, isspmatrix, dok_matrix, lil_matrix, bsr_matrix)
warnings.simplefilter('ignore',SparseEfficiencyWarning)


def deepLearning(feature_matrix, tags_vector, nodeWithTags_to_features, gnx, train_size, graph_name, analysis_type):
    def multiplications(data):
        # A is the adjacency matrix
        # A transpose
        AT = tf.transpose(A)

        # A^2
        AA = tf.matmul(A,A)

        # A transpose ^ 2
        ATAT = tf.transpose(AA)

        # A * (A transpose)
        AAT = tf.matmul(A,AT)

        # (A transpose) * A
        ATA = tf.matmul(AT,A)

        # the multiplications
        AX = tf.matmul(A, data)

        ATX = tf.matmul(AT, data)

        AAX = tf.matmul(AA, data)

        AATX = tf.matmul(AAT, data)

        ATAX = tf.matmul(ATA, data)

        ATATX = tf.matmul(ATAT, data)

        return AX, ATX, AAX, AATX, ATAX, ATATX

    def neural_network_model(data_A, data_AT, data_AA, data_AAT, data_ATA, data_ATAT):
        # the wights and biases of the first layer
        hidden_1_layer = {'weights_A':tf.Variable(tf.random_normal([feature_matrix.shape[1], n_nodes_hl1])),
                          'weights_AT':tf.Variable(tf.random_normal([feature_matrix.shape[1], n_nodes_hl1])),
                          'weights_AA':tf.Variable(tf.random_normal([feature_matrix.shape[1], n_nodes_hl1])),
                          'weights_AAT':tf.Variable(tf.random_normal([feature_matrix.shape[1], n_nodes_hl1])),
                          'weights_ATA':tf.Variable(tf.random_normal([feature_matrix.shape[1], n_nodes_hl1])),
                          'weights_ATAT':tf.Variable(tf.random_normal([feature_matrix.shape[1], n_nodes_hl1])),
                          'biases':tf.Variable(tf.random_normal([6*n_nodes_hl1]))}

        # hidden_2_layer = {'weights':tf.Variable(tf.random_normal([6*n_nodes_hl1, n_nodes_hl2])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

        # the wights and biases of the output layer
        output_layer = {'weights':tf.Variable(tf.random_normal([6*n_nodes_hl1, int(n_classes)])), 'biases':tf.Variable(tf.random_normal([int(n_classes)]))}

        l1_A = tf.matmul(data_A, hidden_1_layer['weights_A'])

        l1_AT = tf.matmul(data_AT, hidden_1_layer['weights_AT'])

        l1_AA = tf.matmul(data_AA, hidden_1_layer['weights_AA'])

        l1_AAT = tf.matmul(data_AAT, hidden_1_layer['weights_AAT'])

        l1_ATA = tf.matmul(data_ATA, hidden_1_layer['weights_ATA'])

        l1_ATAT = tf.matmul(data_ATAT, hidden_1_layer['weights_ATAT'])

        # the concatenation of the multiplications
        l1 = tf.concat([tf.concat([l1_A,l1_AT],1),tf.concat([tf.concat([l1_AA,l1_AAT],1),tf.concat([l1_ATA,l1_ATAT],1)],1)],1)
        l1 = tf.add(l1,hidden_1_layer['biases'])
        if a == 'relu':
            l1 = tf.nn.relu(l1)
        else:
            l1 = tf.nn.elu(l1)

        # l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
        # l2 = tf.nn.elu(l2)
        # l2 = tf.nn.dropout(l2, 0.95)

        output = tf.add(tf.matmul(l1, output_layer['weights']), output_layer['biases'])
        if a == 'relu':
            output = tf.nn.relu(output)
        else:
            output = tf.nn.elu(output)

        if dropout != 1:
            output = tf.nn.dropout(output, dropout)

        return output, hidden_1_layer, l1, output_layer['weights']




    def train_neural_network(x, data_A, data_AT, data_AA, data_AAT, data_ATA, data_ATAT):
        # the multiplications
        ax, atx, aax, aatx, atax, atatx = multiplications(x)
        # the results of the network, the first layer weights, the first layer output, the output layer weights
        prediction, layer1_weights, l1_output, output_weights = neural_network_model(data_A, data_AT, data_AA, data_AAT, data_ATA, data_ATAT)
        weights = tf.trainable_variables()
        # regularization
        if c != 0:
            regularizer = tf.contrib.layers.l1_regularizer(scale=c, scope=None)
            tf.nn.softmax()
            regularization_penalty = tf.contrib.layers.apply_regularization(regularizer, weights)
            # the loss
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Z)) + regularization_penalty
        else:
            # the loss
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Z))
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        hm_epochs = 2000
        batch_size = 100
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # the output files
            # results_file = open(path + '/train results.txt','w')
            loss_file = open(path + '/loss.txt','w')
            accuracy_file = open(path + '/accuracy.txt','w')
            confusion_matrix_file = open(path + '/confusion_matrix.txt','w')
            # weights_file = open(path + '/weights.txt','w')

            # the feature matrix
            batch_x = np.array(feature_matrix, dtype=float)
            # the adjacency matrix
            batch_adj = np.array(adj, dtype=float)
            # batch_adj = batch_adj[:,order]
            a_x, at_x, aa_x, aat_x, ata_x, atat_x = sess.run([ax, atx, aax, aatx, atax, atatx], feed_dict={X: batch_x, A: batch_adj})

            for epoch in range(hm_epochs):

                # the train indexes by the order of the appearance
                train_vec = [i for i in order if train_mask[i]]
                # runs the network
                epoch_loss = 0
                i=0
                while i < int(train_size*len(tags_vector)):
                    start = i
                    end = min([i+batch_size,int(train_size*len(tags_vector))])
                    batch_train_vec = train_vec[start:end]
                    batch_ax = np.array(a_x[batch_train_vec], dtype=float)
                    batch_atx = np.array(at_x[batch_train_vec], dtype=float)
                    batch_aax = np.array(aa_x[batch_train_vec], dtype=float)
                    batch_aatx = np.array(aat_x[batch_train_vec], dtype=float)
                    batch_atax = np.array(ata_x[batch_train_vec], dtype=float)
                    batch_atatx = np.array(atat_x[batch_train_vec], dtype=float)
                    # the classifications matrix
                    batch_y = np.array(tags[batch_train_vec], dtype=float)

                    _, batch_loss, pred, zs, l1_weights, l1_out, output_weight = sess.run([optimizer, cost, prediction, Z, layer1_weights, l1_output, output_weights], feed_dict={Data_A: batch_ax, Data_AT: batch_atx, Data_AA: batch_aax, Data_AAT: batch_aatx, Data_ATA: batch_atax, Data_ATAT:batch_atatx , Z: batch_y})
                    epoch_loss += batch_loss
                    i+=batch_size

                loss_file.write('Epoch ' + str(epoch + 1) + ' completed out of ' + str(hm_epochs) + ' loss: ' + str(epoch_loss) + '\n')

                if (epoch + 1)%100 == 0:
                    # prints the results
                    # weights_file.write('Epoch ' + str(epoch + 1) + ' completed out of ' + str(hm_epochs) + ' loss: ' + str(epoch_loss) + '\n\n')
                    # for atype in ['A', 'AT', 'AA', 'AAT', 'ATA', 'ATAT']:
                    #     weights_file.write(atype+ '\n\n')
                    #     weights_file.write(str(l1_weights['weights_'+atype])+ '\n\n')

                    # weights_file.write('l1 output'+'\n\n')
                    # weights_file.write(str(l1_out)+'\n\n')
                    # weights_file.write('output weights'+'\n\n')
                    # weights_file.write(str(output_weight)+'\n\n')

                    # results_file.write('Epoch ' + str(epoch + 1) + ' completed out of ' + str(hm_epochs) + ' loss: ' + str(epoch_loss) + '\n\n')
                    # for index in range(len(pred)):
                    #     results_file.write(str(pred[index])+'\n')
                    #     results_file.write(str(zs[index])+'\n\n\n')

                    trainLogits=[]
                    trainLabels=[]
                    testLogits=[]
                    testLabels=[]
                    for i in range(len(gnx.nodes())):
                        if train_mask[i]:
                            trainLogits.append(prediction[i])
                            trainLabels.append(Z[i])
                        if test_mask[i]:
                            testLogits.append(prediction[i])
                            testLabels.append(Z[i])
                    # the correctness of the test set predictions (0 or 1 for each node)
                    correct = tf.equal(tf.argmax(testLogits, 1), tf.argmax(testLabels, 1))
                    # the correctness of the train set predictions (0 or 1 for each node)
                    correct_train = tf.equal(tf.argmax(trainLogits, 1), tf.argmax(trainLabels, 1))
                    # confusion matrix
                    confusion_matrix = tf.confusion_matrix(tf.argmax(prediction, 1), tf.argmax(Z, 1))
                    # the average of the correct of the test set (the accuracy)
                    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                    # the average of the correct of the train set (the accuracy)
                    accuracy_train = tf.reduce_mean(tf.cast(correct_train, 'float'))

                    # runs the tests and prints the results
                    # test_x = np.array(feature_matrix, dtype=float)
                    # test_adj = np.array(adj, dtype=float)
                    # test_ax, test_atx, test_aax, test_aatx, test_atax, test_atatx = sess.run([ax, atx, aax, aatx, atax, atatx], feed_dict={X: test_x, A: test_adj})
                    test_y = np.array(tags, dtype=float)

                    cm, ac, ac_train= sess.run([confusion_matrix, accuracy, accuracy_train], feed_dict={Data_A: a_x, Data_AT: at_x, Data_AA: aa_x, Data_AAT: aat_x, Data_ATA: ata_x, Data_ATAT:atat_x , Z: test_y})
                    accuracy_file.write('test accuracy: ' + str(ac) + '\n')
                    accuracy_file.write('train accuracy: ' + str(ac_train) + '\n')
                    confusion_matrix_file.write(str(cm)+'\n#\n')

                if epoch != hm_epochs - 1 and b == 'doShuffle':
                    np.random.shuffle(order)
            # results_file.close()
            loss_file.close()
            accuracy_file.close()
            confusion_matrix_file.close()
            # weights_file.close()

    # True if the node is in the test set
    test_mask = [False for i in range(int(len(gnx.nodes())))]
    # True if the node is in the train set
    train_mask = [True for i in range(int(len(gnx.nodes())))]
    # the classification matrix
    tags = feature_matrix[:,0:-1]
    tags = np.asmatrix(tags)
    for i in range(int(train_size*len(gnx.nodes())),len(gnx.nodes())):
        train_mask[i] = False
        test_mask[i] = True
        tags[i,int(tags_vector[i])] = 1
    # y_train = feature_matrix[:,-1]
    # y_test = feature_matrix[:,-1]
    # for index in range(len(list(tags_vector))):
        # if train_mask[index]:
        #     y_test[index][int(tags_vector[index])] = 0
        # if test_mask[index]:
        #     y_test[index][int(tags_vector[index])] = 1
        # tags[index,int(tags_vector[index])] = 1

    # the adjacency matrix
    adj = nx.adjacency_matrix(gnx, nodeWithTags_to_features)
    # adj.setdiag(adj.diagonal()+1.0)
    adj = adj.todense()
    adj = np.asmatrix(adj, 'float32')
    # ns = []
    # for node in nodeWithTags_to_features:
    #     ns.append(math.sqrt(float(len(gnx.neighbors(node))) + 1.0))
    # for i in range(len(nodeWithTags_to_features)):
    #     for j in range(len(nodeWithTags_to_features)):
    #         adj[i,j] /= (ns[i]*ns[j])

    # number of classes
    n_classes = max(tags_vector) + 1

    #for n_nodes_hl1 in [500,281,158,89,50]:

        #for n_nodes_hl2 in [200,93,43,20]:
        # for n_nodes_hl2 in [93]:
    # for dropout in [1,0.95,0.9,0.98]:
    # for c in [0.005,0,0.1,0.5]:
            # for b in ['doShuffle','doNotShuffle']:
                # for a in ['elu','relu']:
                    # for n_nodes_hl1 in [25,20,10]:
    for dropout in [1]:
        # regularization
        for c in [0]:
            for b in ['doShuffle']:
                for a in ['elu']:
                    for n_nodes_hl1 in [25]:
                        for run in range(5):
                            path = 'C:/Users/Admin/Desktop/galfork/outputs/'+ graph_name+'/'+str(analysis_type) + ', ' + b + ', ' + a + ', ls ' + str([n_nodes_hl1]) + ', 2000 epochs, batch size 100'+', run'+str(run+1)
                            if(not os.path.exists(path)):
                                os.makedirs(path)
                                # the input matrix
                                X = tf.placeholder('float', [len(gnx.nodes()), feature_matrix.shape[1]])

                                # the output
                                Z = tf.placeholder('float')

                                # the adjacency matrix
                                A = tf.placeholder('float', [len(gnx.nodes()), len(gnx.nodes())])

                                # the multiplications
                                Data_A = tf.placeholder('float', [None, feature_matrix.shape[1]])
                                Data_AT = tf.placeholder('float', [None, feature_matrix.shape[1]])
                                Data_AA = tf.placeholder('float', [None, feature_matrix.shape[1]])
                                Data_AAT = tf.placeholder('float', [None, feature_matrix.shape[1]])
                                Data_ATA = tf.placeholder('float', [None, feature_matrix.shape[1]])
                                Data_ATAT = tf.placeholder('float', [None, feature_matrix.shape[1]])

                                # the order of the nodes
                                order = np.arange(feature_matrix.shape[0])

                                train_neural_network(X, Data_A, Data_AT, Data_AA, Data_AAT, Data_ATA, Data_ATAT)








# def deepLearning(feature_matrix, tags_vector, nodeWithTags_to_features, gnx, train_size, graph_name, analysis_type):
#
#     def neural_network_model(data):
#         # the wights and biases of the first layer
#         hidden_1_layer = {'weights_A':tf.Variable(tf.random_normal([feature_matrix.shape[1], n_nodes_hl1])),
#                           'weights_AT':tf.Variable(tf.random_normal([feature_matrix.shape[1], n_nodes_hl1])),
#                           'weights_AA':tf.Variable(tf.random_normal([feature_matrix.shape[1], n_nodes_hl1])),
#                           'weights_AAT':tf.Variable(tf.random_normal([feature_matrix.shape[1], n_nodes_hl1])),
#                           'weights_ATA':tf.Variable(tf.random_normal([feature_matrix.shape[1], n_nodes_hl1])),
#                           'weights_ATAT':tf.Variable(tf.random_normal([feature_matrix.shape[1], n_nodes_hl1])),
#                           'biases':tf.Variable(tf.random_normal([6*n_nodes_hl1]))}
#
#         # hidden_2_layer = {'weights':tf.Variable(tf.random_normal([6*n_nodes_hl1, n_nodes_hl2])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
#
#         # the wights and biases of the output layer
#         output_layer = {'weights':tf.Variable(tf.random_normal([6*n_nodes_hl1, int(n_classes)])), 'biases':tf.Variable(tf.random_normal([int(n_classes)]))}
#
#         # A is the adjacency matrix
#         # A transpose
#         AT = tf.transpose(A)
#
#         # A^2
#         AA = tf.matmul(A,A)
#
#         # A transpose ^ 2
#         ATAT = tf.transpose(AA)
#
#         # A * (A transpose)
#         AAT = tf.matmul(A,AT)
#
#         # (A transpose) * A
#         ATA = tf.matmul(AT,A)
#
#         # the multiplications
#         l1_A = tf.matmul(tf.matmul(A, data), hidden_1_layer['weights_A'])
#
#         l1_AT = tf.matmul(tf.matmul(AT, data), hidden_1_layer['weights_AT'])
#
#         l1_AA = tf.matmul(tf.matmul(AA, data), hidden_1_layer['weights_AA'])
#
#         l1_AAT = tf.matmul(tf.matmul(AAT, data), hidden_1_layer['weights_AAT'])
#
#         l1_ATA = tf.matmul(tf.matmul(ATA, data), hidden_1_layer['weights_ATA'])
#
#         l1_ATAT = tf.matmul(tf.matmul(ATAT, data), hidden_1_layer['weights_ATAT'])
#
#         # the concatenation of the multiplications
#         l1 = tf.concat([tf.concat([l1_A,l1_AT],1),tf.concat([tf.concat([l1_AA,l1_AAT],1),tf.concat([l1_ATA,l1_ATAT],1)],1)],1)
#         l1 = tf.add(l1,hidden_1_layer['biases'])
#         if a == 'relu':
#             l1 = tf.nn.relu(l1)
#         else:
#             l1 = tf.nn.elu(l1)
#
#         # l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
#         # l2 = tf.nn.elu(l2)
#         # l2 = tf.nn.dropout(l2, 0.95)
#
#         output = tf.add(tf.matmul(l1, output_layer['weights']), output_layer['biases'])
#         if a == 'relu':
#             output = tf.nn.relu(output)
#         else:
#             output = tf.nn.elu(output)
#
#         if dropout != 1:
#             output = tf.nn.dropout(output, dropout)
#
#         return tf.matmul(Change_matrix, output), hidden_1_layer, l1, output_layer['weights']
#
#     def train_neural_network(x):
#         # the results of the network, the first layer weights, the first layer output, the output layer weights
#         prediction, layer1_weights, l1_output, output_weights = neural_network_model(x)
#         # the prediction of the train set
#         trainLogits = []
#         # the real tags of the train set
#         trainLabels = []
#         for i in range(len(gnx.nodes())):
#             if train_mask[i]:
#                 trainLogits.append(prediction[i])
#                 trainLabels.append(Z[i])
#         weights = tf.trainable_variables()
#         # regularization
#         if c != 0:
#             regularizer = tf.contrib.layers.l1_regularizer(scale=c, scope=None)
#             tf.nn.softmax()
#             regularization_penalty = tf.contrib.layers.apply_regularization(regularizer, weights)
#             # the loss
#             cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=trainLogits, labels=trainLabels)) + regularization_penalty
#         else:
#             # the loss
#             cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=trainLogits, labels=trainLabels))
#         optimizer = tf.train.AdamOptimizer().minimize(cost)
#
#         hm_epochs = 2000
#         with tf.Session() as sess:
#             sess.run(tf.global_variables_initializer())
#             # the output files
#             results_file = open(path + '/train results.txt','w')
#             loss_file = open(path + '/loss.txt','w')
#             accuracy_file = open(path + '/accuracy.txt','w')
#             confusion_matrix_file = open(path + '/confusion_matrix.txt','w')
#             weights_file = open(path + '/weights.txt','w')
#             for epoch in range(hm_epochs):
#                 # helps to return the result matrix to the original order (so the train set is in the beginning again)
#                 index_for_change_matrix = [np.argwhere(order==i)[0][0] for i in range(len(gnx.nodes()))]
#                 # the feature matrix
#                 batch_x = np.array(feature_matrix[order], dtype=float)
#                 # the classifications matrix
#                 batch_y = np.array(tags, dtype=float)
#                 # the adjacency matrix
#                 batch_adj = np.array(adj[order], dtype=float)
#                 batch_adj = batch_adj[:,order]
#                 # this matrix returns the result matrix to the original order (so the train set is in the beginning again)
#                 batch_change_matrix = np.array(change_matrix[index_for_change_matrix], dtype=float)
#                 # runs the network
#                 _, epoch_loss, pred, zs, l1_weights, l1_out, output_weight = sess.run([optimizer, cost, prediction, Z, layer1_weights, l1_output, output_weights], feed_dict={X: batch_x, Z: batch_y, A: batch_adj, Change_matrix: batch_change_matrix})
#                 loss_file.write('Epoch ' + str(epoch + 1) + ' completed out of ' + str(hm_epochs) + ' loss: ' + str(epoch_loss) + '\n')
#
#                 if (epoch + 1)%100 == 0:
#                     # prints the results
#                     weights_file.write('Epoch ' + str(epoch + 1) + ' completed out of ' + str(hm_epochs) + ' loss: ' + str(epoch_loss) + '\n\n')
#                     for atype in ['A', 'AT', 'AA', 'AAT', 'ATA', 'ATAT']:
#                         weights_file.write(atype+ '\n\n')
#                         weights_file.write(str(l1_weights['weights_'+atype])+ '\n\n')
#
#                     weights_file.write('l1 output'+'\n\n')
#                     weights_file.write(str(l1_out)+'\n\n')
#                     weights_file.write('output weights'+'\n\n')
#                     weights_file.write(str(output_weight)+'\n\n')
#
#                     results_file.write('Epoch ' + str(epoch + 1) + ' completed out of ' + str(hm_epochs) + ' loss: ' + str(epoch_loss) + '\n\n')
#                     for index in range(len(pred)):
#                         results_file.write(str(pred[index])+'\n')
#                         results_file.write(str(zs[index])+'\n\n\n')
#
#                     # the prediction of the test set
#                     testLogits = []
#                     # the real tags of the test set
#                     testLabels = []
#                     for i in range(len(gnx.nodes())):
#                         if test_mask[i]:
#                             testLogits.append(prediction[i])
#                             testLabels.append(Z[i])
#                     # the correctness of the test set predictions (0 or 1 for each node)
#                     correct = tf.equal(tf.argmax(testLogits, 1), tf.argmax(testLabels, 1))
#                     # the correctness of the train set predictions (0 or 1 for each node)
#                     correct_train = tf.equal(tf.argmax(trainLogits, 1), tf.argmax(trainLabels, 1))
#                     # confusion matrix
#                     confusion_matrix = tf.confusion_matrix(tf.argmax(testLogits, 1), tf.argmax(testLabels, 1))
#                     # the average of the correct of the test set (the accuracy)
#                     accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
#                     # the average of the correct of the train set (the accuracy)
#                     accuracy_train = tf.reduce_mean(tf.cast(correct_train, 'float'))
#                     # runs the tests and prints the results
#                     test_x = np.array(feature_matrix, dtype=float)
#                     test_y = np.array(tags, dtype=float)
#                     test_adj = np.array(adj, dtype=float)
#                     test_change_matrix = np.array(change_matrix, dtype=float)
#                     cm, ac, ac_train= sess.run([confusion_matrix, accuracy, accuracy_train], feed_dict={X: test_x, Z: test_y, A: test_adj, Change_matrix: test_change_matrix})
#                     accuracy_file.write('test accuracy: ' + str(ac) + '\n')
#                     accuracy_file.write('train accuracy: ' + str(ac_train) + '\n')
#                     confusion_matrix_file.write(str(cm)+'\n#\n')
#
#                 if epoch != hm_epochs - 1 and b == 'doShuffle':
#                     np.random.shuffle(order)
#             results_file.close()
#             loss_file.close()
#             accuracy_file.close()
#             confusion_matrix_file.close()
#             weights_file.close()
#
#     # True if the node is in the test set
#     test_mask = [False for i in range(int(len(gnx.nodes())))]
#     # True if the node is in the train set
#     train_mask = [True for i in range(int(len(gnx.nodes())))]
#     # the classification matrix
#     tags = feature_matrix[:,0:-1]
#     tags = np.asmatrix(tags)
#     for i in range(int(train_size*len(gnx.nodes())),len(gnx.nodes())):
#         train_mask[i] = False
#         test_mask[i] = True
#         tags[i,int(tags_vector[i])] = 1
#     # y_train = feature_matrix[:,-1]
#     # y_test = feature_matrix[:,-1]
#     # for index in range(len(list(tags_vector))):
#         # if train_mask[index]:
#         #     y_test[index][int(tags_vector[index])] = 0
#         # if test_mask[index]:
#         #     y_test[index][int(tags_vector[index])] = 1
#         # tags[index,int(tags_vector[index])] = 1
#
#     # the adjacency matrix
#     adj = nx.adjacency_matrix(gnx, nodeWithTags_to_features)
#     # adj.setdiag(adj.diagonal()+1.0)
#     adj = adj.todense()
#     adj = np.asmatrix(adj, 'float32')
#     # ns = []
#     # for node in nodeWithTags_to_features:
#     #     ns.append(math.sqrt(float(len(gnx.neighbors(node))) + 1.0))
#     # for i in range(len(nodeWithTags_to_features)):
#     #     for j in range(len(nodeWithTags_to_features)):
#     #         adj[i,j] /= (ns[i]*ns[j])
#
#     # this matrix returns the result matrix to the original order (so the train set is in the beginning again)
#     # for start it is the identity matrix
#     change_matrix = [[0 for i in range(len(gnx.nodes()))] for j in range(len(gnx.nodes()))]
#     for i in range(len(gnx.nodes())):
#         change_matrix[i][i] = 1
#     change_matrix = np.asmatrix(change_matrix, 'float32')
#
#     # number of classes
#     n_classes = max(tags_vector) + 1
#
#     #for n_nodes_hl1 in [500,281,158,89,50]:
#
#         #for n_nodes_hl2 in [200,93,43,20]:
#         # for n_nodes_hl2 in [93]:
#     # for dropout in [1,0.95,0.9,0.98]:
#     # for c in [0.005,0,0.1,0.5]:
#             # for b in ['doShuffle','doNotShuffle']:
#                 # for a in ['elu','relu']:
#                     # for n_nodes_hl1 in [25,20,10]:
#     for dropout in [1]:
#         # regularization
#         for c in [0]:
#             for b in ['doShuffle']:
#                 for a in ['elu']:
#                     for n_nodes_hl1 in [25]:
#                         for run in range(5):
#                             path = 'C:/Users/Admin/Desktop/galfork/outputs/'+ graph_name+'/'+str(analysis_type) + ', ' + b + ', ' + a + ', ls ' + str([n_nodes_hl1]) + ', 2000 epochs'+', tr' + str(train_size)+', run' + str(run+1)
#                             if(not os.path.exists(path)):
#                                 os.makedirs(path)
#                                 # the input matrix
#                                 X = tf.placeholder('float', [len(gnx.nodes()), feature_matrix.shape[1]])
#
#                                 # the output
#                                 Z = tf.placeholder('float')
#
#                                 # the adjacency matrix
#                                 A = tf.placeholder('float', [len(gnx.nodes()), len(gnx.nodes())])
#
#                                 # this matrix returns the result matrix to the original order (so the train set is in the beginning again)
#                                 Change_matrix = tf.placeholder('float', [len(gnx.nodes()), len(gnx.nodes())])
#
#                                 # the order of the nodes
#                                 order = np.arange(feature_matrix.shape[0])
#
#                                 train_neural_network(X)
#
#
#









    # tags_dist = [float(0) for _ in range(int(max(tags_vector))+1)]
    # for node_index in range(len(tags_vector)):
    #     tags_dist[int(tags_vector[node_index])] += 1 / len(tags_vector)
    # print(tags_dist)
    #
    # x=range(int(max(tags_vector))+1)
    # plt.bar(x, height= tags_dist)
    # plt.xticks([float(i) for i in x], x)
    # plt.xlabel('class')
    # plt.ylabel('probability')
    # plt.title(graph_name+' classes probability')
    #
    #
    # plt.show()


# def deepLearning(feature_matrix, train_tags, test_tags, x_train, x_test, gnx, train_size, graph_name, analysis_type):
#     # number of classes
#     n_classes = max(train_tags) + 1
#     print(n_classes)
#
#     tr_tags = [[0 for i in range(int(n_classes))] for j in range(len(train_tags))]
#     tr_tags = np.asmatrix(tr_tags)
#     for index in range(len(train_tags)):
#         tr_tags[index,int(train_tags[index])] = 1
#
#     te_tags = [[0 for i in range(int(n_classes))] for j in range(len(test_tags))]
#     te_tags = np.asmatrix(te_tags)
#     for index in range(len(test_tags)):
#         te_tags[index,int(test_tags[index])] = 1
#
#     n_nodes_hl1 = 100
#     n_nodes_hl2 = 50
#     n_nodes_hl3 = 20
#     elu = 'elu'
#     reg = 0.005
#     dropout = 0.95
#     hm_epochs = 2000
#     batch_size = 100
#     path = 'C:/Users/Admin/Desktop/galfork/outputs/'+ graph_name+'/'+str(analysis_type) + ', ' + elu + ', reg ' + str(reg) + ', ls ' + str([n_nodes_hl1,n_nodes_hl2,n_nodes_hl3])+ ', drop ' + str(dropout) + ', ' + str(hm_epochs) + ' epochs, batch ' + str(batch_size)
#     if(not os.path.exists(path)):
#         os.makedirs(path)
#         # the input matrix
#         X = tf.placeholder('float', [None, feature_matrix.shape[1]])
#
#         # the output
#         Z = tf.placeholder('float')
#
#         def neural_network_model(data):
#             hidden_1_layer = {'weights':tf.Variable(tf.random_normal([feature_matrix.shape[1], n_nodes_hl1])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
#
#             hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
#
#             hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
#
#             output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, int(n_classes)])), 'biases':tf.Variable(tf.random_normal([int(n_classes)]))}
#
#             l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
#             if elu == 'relu':
#                 l1 = tf.nn.relu(l1)
#             else:
#                 l1 = tf.nn.elu(l1)
#
#             l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
#             if elu == 'relu':
#                 l2 = tf.nn.relu(l2)
#             else:
#                 l2 = tf.nn.elu(l2)
#
#             l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
#             if elu == 'relu':
#                 l3 = tf.nn.relu(l3)
#             else:
#                 l3 = tf.nn.elu(l3)
#
#             l3 = tf.nn.dropout(l3, dropout)
#
#             output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])
#
#             return output
#
#         def train_neural_network(x):
#             prediction = neural_network_model(x)
#             weights = tf.trainable_variables()
#             if reg != 0:
#                 regularizer = tf.contrib.layers.l1_regularizer(scale=reg, scope=None)
#                 regularization_penalty = tf.contrib.layers.apply_regularization(regularizer, weights)
#                 cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Z)) + regularization_penalty
#             else:
#                 cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Z))
#             optimizer = tf.train.AdamOptimizer().minimize(cost)
#
#             train_predictions = []
#             train_labels = []
#
#             with tf.Session() as sess:
#                 sess.run(tf.global_variables_initializer())
#                 loss_file = open(path + '/loss.txt','w')
#                 accuracy_file = open(path + '/accuracy.txt','w')
#                 confusion_matrix_file = open(path + '/confusion_matrix.txt','w')
#                 for epoch in range(hm_epochs):
#                     epoch_loss = float(0)
#                     i=0
#                     while i < len(x_train):
#                         start = i
#                         end = min([len(x_train) - 1,i+batch_size])
#                         batch_x = np.array(x_train[start:end])
#                         batch_y = np.array(tr_tags[start:end])
#
#                         _, batch_loss, pred, label = sess.run([optimizer, cost, prediction, Z], feed_dict={X: batch_x, Z: batch_y})
#                         epoch_loss += batch_loss
#                         # train_predictions.append(pred)
#                         # train_labels.append(label)
#                         epoch_loss += batch_loss
#                         i+=batch_size
#                     loss_file.write('Epoch ' + str(epoch + 1) + ' completed out of ' + str(hm_epochs) + ' loss: ' + str(epoch_loss) + '\n')
#                     if (epoch + 1)%100 == 0:
#                         correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(Z, 1))
#                         #     correct_train = tf.equal(tf.argmax(train_predictions, 1), tf.argmax(train_labels, 1))
#                         confusion_matrix = tf.confusion_matrix(tf.argmax(prediction, 1), tf.argmax(Z, 1))
#                         accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
#                         #     accuracy_train = tf.reduce_mean(tf.cast(correct_train, 'float'))
#                         test_x = np.array(x_test, dtype=float)
#                         test_y = np.array(te_tags, dtype=float)
#                         #     cm, ac, ac_train = sess.run([confusion_matrix, accuracy, accuracy_train], feed_dict={X: test_x, Z: test_y})
#                         cm, ac = sess.run([confusion_matrix, accuracy], feed_dict={X: test_x, Z: test_y})
#                         accuracy_file.write('test accuracy: ' + str(ac) + '\n')
#                         #     accuracy_file.write('train accuracy: ' + str(ac_train) + '\n')
#                         confusion_matrix_file.write(str(cm)+'\n#\n')
#                 loss_file.close()
#                 accuracy_file.close()
#                 confusion_matrix_file.close()
#         train_neural_network(X)
