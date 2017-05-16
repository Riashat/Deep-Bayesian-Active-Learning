from __future__ import print_function
import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import random
random.seed(5001)


def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test

"""
# This creates an MLP of two hidden layers of 800 units each, followed by
# a softmax output layer of 10 units. It applies 20% dropout to the input
# data and 50% dropout to the hidden layers.
"""
def build_mlp(input_var=None):

    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)

    # Apply 20% dropout to the input data: 
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.25)

    ## Dropout per kernel - dropout full channel of feature maps
    ## same as SpatialDropout2D in keras
    # l_in_drop = lasagne.layers.spatial_dropout(l_in)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.25)
    # l_hid1_drop = lasagne.layers.spatial_dropout(l_hid1)


    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.25)
    #l_hid2_drop = lasagne.layers.spatial_dropout(l_hid2)

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]




def split_train_pool_data(X_train, y_train, X_val, y_val):

    random_split = np.asarray(random.sample(range(0,X_train.shape[0]), X_train.shape[0]))

    X_train = X_train[random_split, :, :, :]
    y_train = y_train[random_split]

    X_valid = X_train[10000:15000, :, :, :]
    y_valid = y_train[10000:15000]

    X_pool = X_train[20000:60000, :, :, :]
    y_pool = y_train[20000:60000]


    X_train = X_train[0:10000, :, :, :]
    y_train = y_train[0:10000]


    return X_train, y_train, X_valid, y_valid, X_pool, y_pool


def get_initial_training_data(X_train_All, y_train_All):
    #training data to have equal distribution of classes
    idx_0 = np.array( np.where(y_train_All==0)  ).T
    idx_0 = idx_0[0:10,0]
    X_0 = X_train_All[idx_0, :, :, :]
    y_0 = y_train_All[idx_0]

    idx_1 = np.array( np.where(y_train_All==1)  ).T
    idx_1 = idx_1[0:10,0]
    X_1 = X_train_All[idx_1, :, :, :]
    y_1 = y_train_All[idx_1]

    idx_2 = np.array( np.where(y_train_All==2)  ).T
    idx_2 = idx_2[0:10,0]
    X_2 = X_train_All[idx_2, :, :, :]
    y_2 = y_train_All[idx_2]

    idx_3 = np.array( np.where(y_train_All==3)  ).T
    idx_3 = idx_3[0:10,0]
    X_3 = X_train_All[idx_3, :, :, :]
    y_3 = y_train_All[idx_3]

    idx_4 = np.array( np.where(y_train_All==4)  ).T
    idx_4 = idx_4[0:10,0]
    X_4 = X_train_All[idx_4, :, :, :]
    y_4 = y_train_All[idx_4]

    idx_5 = np.array( np.where(y_train_All==5)  ).T
    idx_5 = idx_5[0:10,0]
    X_5 = X_train_All[idx_5, :, :, :]
    y_5 = y_train_All[idx_5]

    idx_6 = np.array( np.where(y_train_All==6)  ).T
    idx_6 = idx_6[0:10,0]
    X_6 = X_train_All[idx_6, :, :, :]
    y_6 = y_train_All[idx_6]

    idx_7 = np.array( np.where(y_train_All==7)  ).T
    idx_7 = idx_7[0:10,0]
    X_7 = X_train_All[idx_7, :, :, :]
    y_7 = y_train_All[idx_7]

    idx_8 = np.array( np.where(y_train_All==8)  ).T
    idx_8 = idx_8[0:10,0]
    X_8 = X_train_All[idx_8, :, :, :]
    y_8 = y_train_All[idx_8]

    idx_9 = np.array( np.where(y_train_All==9)  ).T
    idx_9 = idx_9[0:10,0]
    X_9 = X_train_All[idx_9, :, :, :]
    y_9 = y_train_All[idx_9]

    X_train = np.concatenate((X_0, X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8, X_9), axis=0 )
    y_train = np.concatenate((y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9), axis=0 )


    return X_train, y_train




def train_model(num_epochs, X_train, y_train, X_val, y_val, train_fn, val_fn):

    print("Starting training...")

    for epoch in range(num_epochs):

        train_err = 0
        train_batches = 0
        start_time = time.time()

        for batch in iterate_minibatches(X_train, y_train, 64, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        training_error = train_err / train_batches


        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0


        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):

            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        validation_error = val_err / val_batches
    
    return training_error, validation_error


def test_model(X_test, y_test, val_fn):

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1

    test_error = test_err / test_batches
    test_accuracy = test_acc / test_batches


    return test_error, test_accuracy
    # print("Final results:")
    # print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    # print("  test accuracy:\t\t{:.2f} %".format(
    #     test_acc / test_batches * 100))




def active_learning(model, num_epochs, acquisition_iterations, nb_classes):

    all_accuracy = 0
    all_error = 0
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    X_train, y_train, X_val, y_val, X_pool, y_pool = split_train_pool_data(X_train, y_train, X_val, y_val)
    X_train, y_train = get_initial_training_data(X_train, y_train)

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    print("Building model and compiling functions...")
    model == 'mlp'

    network = build_mlp(input_var)

    prediction = lasagne.layers.get_output(network)

    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)

    updates = lasagne.updates.nesterov_momentum(
                loss, params, learning_rate=0.01, momentum=0.9)


    """
    For stochastic forward passes
    """
    test_prediction = lasagne.layers.get_output(network, deterministic=False)

    """
    For deterministic passes
    """
    # test_prediction = lasagne.layers.get_output(network, deterministic=True)

    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)   

    test_loss = test_loss.mean()

    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)


    #for evaluating validation loss and accuracy
    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    get_preds = theano.function([input_var], test_prediction)


    """
    Train the model
    """
    training_error, validation_error = train_model(num_epochs, X_train, y_train, X_val, y_val, train_fn, val_fn)

    """
    Evaluate the model
    """
    test_error, test_accuracy = test_model(X_test, y_test, val_fn)

    print ("Test Accuracy", test_accuracy)

    all_accuracy = test_accuracy
    all_error = test_error

    dropout_iterations = 100
    Queries = 10


    for i in range(acquisition_iterations):
        
        print('POOLING ITERATION', i)

        pool_subset = 2000
        pool_subset_dropout = np.asarray(random.sample(range(0,X_pool.shape[0]), pool_subset))
        X_pool_Dropout = X_pool[pool_subset_dropout, :, :, :]
        y_pool_Dropout = y_pool[pool_subset_dropout]

        score_All = np.zeros(shape=(X_pool_Dropout.shape[0], nb_classes))
        All_Entropy_Dropout = np.zeros(shape=X_pool_Dropout.shape[0])

        all_dropout_classes = np.zeros(shape=(X_pool_Dropout.shape[0], dropout_iterations))


        print ("MC Dropout Iterations")
        for d in range(dropout_iterations):


            dropout_score = get_preds(X_pool_Dropout)
            score_All = score_All + dropout_score

            dropout_score_log = np.log2(dropout_score)
            Entropy_Compute = - np.multiply(dropout_score, dropout_score_log)
            Entropy_Per_Dropout = np.sum(Entropy_Compute, axis=1)

            All_Entropy_Dropout = All_Entropy_Dropout + Entropy_Per_Dropout

            """
            For observing dropout uncertainty over images
            """
            #save in 
            dropout_classes = np.max(dropout_score, axis=1)
            all_dropout_classes[:, d] = dropout_classes


        ### for plotting uncertainty
        predicted_class = np.max(all_dropout_classes, axis=1)
        predicted_class_std = np.std(all_dropout_classes, axis=1)

        # np.save('/Users/Riashat/Documents/PhD_Research/Bayesian_DNNs/BayesianHypernet/Active_Learning_Tasks/MC_Dropout/dropout_uncertainty/predicted_class.npy', predicted_class)
        # np.save('/Users/Riashat/Documents/PhD_Research/Bayesian_DNNs/BayesianHypernet/Active_Learning_Tasks/MC_Dropout/dropout_uncertainty/predicted_class_std.npy', predicted_class_std)
        

        Avg_Pi = np.divide(score_All, dropout_iterations)
        Log_Avg_Pi = np.log2(Avg_Pi)
        Entropy_Avg_Pi = - np.multiply(Avg_Pi, Log_Avg_Pi)
        Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)

        G_X = Entropy_Average_Pi

        Average_Entropy = np.divide(All_Entropy_Dropout, dropout_iterations)
        F_X = Average_Entropy
        U_X = G_X - F_X

        sort_values = U_X.flatten()
        x_pool_index = sort_values.argsort()[-Queries:][::-1]


        Pooled_X = X_pool_Dropout[x_pool_index, :,:,:]
        Pooled_Y = y_pool_Dropout[x_pool_index] 


        #first delete the random subset used for test time dropout from X_Pool
        #Delete the pooled point from this pool set (this random subset)
        #then add back the random pool subset with pooled points deleted back to the X_Pool set
        delete_Pool_X = np.delete(X_pool, (pool_subset_dropout), axis=0)
        delete_Pool_Y = np.delete(y_pool, (pool_subset_dropout), axis=0)        

        delete_Pool_X_Dropout = np.delete(X_pool_Dropout, (x_pool_index), axis=0)
        delete_Pool_Y_Dropout = np.delete(y_pool_Dropout, (x_pool_index), axis=0)
 

        X_pool = np.concatenate((X_pool, X_pool_Dropout), axis=0)
        y_pool = np.concatenate((y_pool, y_pool_Dropout), axis=0)


        X_train = np.concatenate((X_train, Pooled_X), axis=0)
        y_train = np.concatenate((y_train, Pooled_Y), axis=0)           


        print ("Training Data Size", X_train.shape)
        """
        Train and Test with the new training data
        """
        training_error, validation_error = train_model(num_epochs, X_train, y_train, X_val, y_val, train_fn, val_fn)
        test_error, test_accuracy = test_model(X_test, y_test, val_fn)

        print ("Test Accuracy", test_accuracy)

        all_accuracy = np.append(all_accuracy, test_accuracy)
        all_error = np.append(all_error, test_error)



    return all_accuracy


def main():

    num_experiments = 3
    model='mlp'
    num_epochs=50
    acquisition_iterations=98
    nb_classes=10
    
    all_accuracy = np.zeros(shape=(acquisition_iterations+1, num_experiments))

    for i in range(num_experiments):

        accuracy = active_learning(model, num_epochs, acquisition_iterations, nb_classes)

        all_accuracy[:, i] = accuracy 

    mean_accuracy = np.mean(all_accuracy)  

    np.save('dropout_bald_all_accuracy.npy', all_accuracy)
    np.save('dropout_bald_mean_accuracy.npy',mean_accuracy)    


if __name__ == '__main__':

    main()




