import os
import numpy as np
import tensorflow as tf
import math
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense
from model import CNN_HS
from function import BF_FPT_RELIABILITY, AF_FPT_RELIABILITY, PREPARE_DATASET, PREPARE_TRAINING_DATA, F1_SCORE, HS_DETERMINE, PLOT_RESULT, PREPARE_TEST_DATA, PLOT_TSNE, FIND_FPT
from config import EPOCHS, TRAINING_DIR, TEST_DIR, RMS_RESULT, AE_RESULT,BATCH_SIZE, LEANING_RATE

""" 
gpu setting
Set the GPU settings according to the PC environment.
"""

if __name__ == '__main__':
    
    """"""""""""""""""""" TRAINING PROCESS """""""""""""""""""""

    training_data = []
    training_label = []
    
    for train_num, DIR in enumerate(TRAINING_DIR):
        # Convert raw data of 2 channels to NSP image and create label according to ratio
        data, label = PREPARE_TRAINING_DATA(DIR)

        training_data.append(data)
        training_label.append(label)
    
    # Combine data and label to be used for training 
    train_len = len(training_data)
    train_dataset = PREPARE_DATASET(training_data, training_label)
    
    # create CNN_HS model
    model = CNN_HS()
    model.build(input_shape=(None,128,128,3))

    # define loss and optimizer
    train_summary_writer = tf.summary.create_file_writer('Please enter the storage path of log file.')
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEANING_RATE)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')


    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(y_true=labels, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(y_true=labels, y_pred=predictions)
        tf.summary.scalar('loss', train_loss.result(), step=optimizer.iterations)
        tf.summary.scalar('accuracy', train_accuracy.result(), step=optimizer.iterations)


    # start training
    with train_summary_writer.as_default():    
        for epoch in range(EPOCHS):
            step = 0
            for images, labels in train_dataset:
                step += 1
                train_step(images, labels)
                print("RATIO: {}, Epoch: {}/{}, step: {}/{}, loss: {:.5f}, RMSE: {:.5f}".format(1,epoch + 1, EPOCHS, step, math.ceil(train_len / BATCH_SIZE), train_loss.result(), train_accuracy.result()))                                                            
            model.save_weights('save_model/cnn_hs.h5')

 

    """"""""""""""""""""" TEST PROCESS """""""""""""""""""""

    # load CNN-HS model weight
    model = CNN_HS()
    model.build(input_shape=(None,128,128,3))
    model.load_weights('save_model/cnn_hs.h5')

    for test_num, DIR in enumerate(TEST_DIR):
        
        # Convert raw data of 2 channels to NSP image
        test_data = PREPARE_TEST_DATA(DIR)
        
        # Predict HS through NSP image of test data, plot predict results and calculate FPT, Before FPT reliability, After FPT reliability, f1 score
        predict = model.predict(test_data)
        result = HS_DETERMINE(predict)
        bf_results = BF_FPT_RELIABILITY(result)
        af_results = AF_FPT_RELIABILITY(result)
        PLOT_RESULT(result,test_num)
        fpt = FIND_FPT(result)
        F1_SCORE(bf_results,af_results)
        
        # Feature visulaization using t-SNE
        feature = model.FeatureExtraction(test_data)
        PLOT_TSNE(feature,result,RMS_RESULT[test_num],AE_RESULT[test_num],fpt, test_num)




        

    







                                                                                                                                     

            

