from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import pickle
import scipy as sp
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold 
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import csv
import sys
from random import *
from operator import itemgetter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import math
import random
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten


#1DCNN function
def CNN1D_evaluate_model(trainX, testX, trainy, testy, testy_Ori):
    print("____________1DCNN______________")
    verbose, epochs, batch_size = 0, 100, 32
    print("trainX, trainy, testX, testy:", trainX.shape, trainy.shape, testX.shape, testy.shape)
    print("trainX.shape[1], trainy.shape[1]:", trainX.shape[1], trainy.shape[1])
    #n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    #n_features, n_outputs = trainX.shape[1], trainy.shape[1]
    trainX = trainX.values.reshape(trainX.shape[0], trainX.shape[1],1)
    testX = testX.values.reshape(testX.shape[0], testX.shape[1],1)
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    print("-------------after_testX--------------")
    model = Sequential()
    model.add(Conv1D(filters=20, kernel_size=5, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(filters=40, kernel_size=5, activation='relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    #model.add(Dense(1024, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)

    Y_pred = model.predict_classes(testX)
    Y_probs = model.predict(testX)
    #Y_pred = Y_pred + 1

   
    Y_pred=to_categorical(Y_pred,num_classes=4)
   

  
    
    #print("Y_pred, y[validation]", Y_pred, testy_Ori)
    allScores = classification_report(testy_Ori, Y_pred, labels=[0, 1, 2, 3])
    allScores = classification_report(testy_Ori, Y_pred)
    print('allScores:', str(allScores))
    # accuracy: (tp + tn) / (p + n)
    accuracyVal = accuracy_score(testy_Ori, Y_pred)
    print('Accuracy: %f' % accuracyVal)
    
    return accuracy, allScores, accuracyVal
	
	
###Sample copy by STAGE through GAN #######
for stage_number in range(4):
	print("stage :",(stage_number+1))
	for ith in range(30):
		print(str(ith)+"th")
		sample_name = sys.argv[1]	#expr data
		feature_name = sys.argv[2] # feature index
		stage_def = (stage_number+1)

		############Gene index result data loading part#################
		feat = "./" + feature_name
		mutation_intersection_gene_index=np.load(feat)


		###########Original data loading part######################
		BT= './' + sample_name
		with open(BT,'r') as fin:
			lines=fin.readlines()

		samples = lines[0].rstrip().split('\t')[1:]
		genes=[]
		data=[]

		for line in lines[1:]:
			tmp= line.rstrip().split('\t')
			genes.append(tmp[0])
			data.append(tmp[1:])

		samples_number=data[len(genes)-1]

		labels=np.array(samples_number).astype(np.int64)


		data=np.array(data).T


		new_data=[]
		for i in range(len(data)):
			new_data.append(data[i][:-1])



		array_new_data=np.array(new_data)


		one_hot=np.zeros((len(labels),5))

		one_hot[np.arange(len(labels)),labels]=1

		new_one_hot=[]
		for i in range(len(one_hot)):
			new_one_hot.append(one_hot[i][1:])


		new_one_hot=np.array(new_one_hot)
		stage_1_index=[]
		stage_2_index=[]
		stage_3_index=[]
		stage_4_index=[]
		for i in range(len(labels)):
			if labels[i]==1:
				stage_1_index.append(i)
			if labels[i]==2:
				stage_2_index.append(i)
			if labels[i]==3:
				stage_3_index.append(i)
			if labels[i]==4:
				stage_4_index.append(i)
			 
		stage_1_samples=[]
		for i in range(len(stage_1_index)):
			stage_1_samples.append(samples[stage_1_index[i]])
		stage_2_samples=[]
		for i in range(len(stage_2_index)):
			stage_2_samples.append(samples[stage_2_index[i]])
		stage_3_samples=[]
		for i in range(len(stage_3_index)):
			stage_3_samples.append(samples[stage_3_index[i]])
		stage_4_samples=[]
		for i in range(len(stage_4_index)):
			stage_4_samples.append(samples[stage_4_index[i]])

		stage_1_data=[]
		for i in range(len(stage_1_index)):
			stage_1_data.append(new_data[stage_1_index[i]])
		stage_2_data=[]
		for i in range(len(stage_2_index)):
			stage_2_data.append(new_data[stage_2_index[i]])
		stage_3_data=[]
		for i in range(len(stage_3_index)):
			stage_3_data.append(new_data[stage_3_index[i]])
		stage_4_data=[]
		for i in range(len(stage_4_index)):
			stage_4_data.append(new_data[stage_4_index[i]])
		#Data set that collects the entire data for each stage 
		stage_1_data=np.array(stage_1_data,dtype=np.float32)
		stage_2_data=np.array(stage_2_data,dtype=np.float32)
		stage_3_data=np.array(stage_3_data,dtype=np.float32)
		stage_4_data=np.array(stage_4_data,dtype=np.float32)

		#feature selection data
		featured_stage_1_data=[]
		total_featured_stage1_data=[]
		for i in range(stage_1_data.shape[0]):
			tmp=stage_1_data[i]
			for j in range(len(mutation_intersection_gene_index)):
				line_tmp=tmp[mutation_intersection_gene_index[j]]
				featured_stage_1_data.append(line_tmp)
				line_tmp=[]
			total_featured_stage1_data.append(featured_stage_1_data)
			featured_stage_1_data=[]
		  
		featured_stage_2_data=[]
		total_featured_stage2_data=[]
		for i in range(stage_2_data.shape[0]):
			tmp=stage_2_data[i]
			for j in range(len(mutation_intersection_gene_index)):
				line_tmp=tmp[mutation_intersection_gene_index[j]]
				featured_stage_2_data.append(line_tmp)
				line_tmp=[]
			total_featured_stage2_data.append(featured_stage_2_data)
			featured_stage_2_data=[]

		featured_stage_3_data=[]
		total_featured_stage3_data=[]
		for i in range(stage_3_data.shape[0]):
			tmp=stage_3_data[i]
			for j in range(len(mutation_intersection_gene_index)):
				line_tmp=tmp[mutation_intersection_gene_index[j]]
				featured_stage_3_data.append(line_tmp)
				line_tmp=[]
			total_featured_stage3_data.append(featured_stage_3_data)
			featured_stage_3_data=[]
		  
		featured_stage_4_data=[]
		total_featured_stage4_data=[]
		for i in range(stage_4_data.shape[0]):
			tmp=stage_4_data[i]
			for j in range(len(mutation_intersection_gene_index)):
				line_tmp=tmp[mutation_intersection_gene_index[j]]
				featured_stage_4_data.append(line_tmp)
				ine_tmp=[]
			total_featured_stage4_data.append(featured_stage_4_data)
			featured_stage_4_data=[]
		  
		array_total_featured_stage1_data=np.array(total_featured_stage1_data)
		array_total_featured_stage2_data=np.array(total_featured_stage2_data)
		array_total_featured_stage3_data=np.array(total_featured_stage3_data)
		array_total_featured_stage4_data=np.array(total_featured_stage4_data)


		stage_1_data=array_total_featured_stage1_data
		stage_2_data=array_total_featured_stage2_data
		stage_3_data=array_total_featured_stage3_data
		stage_4_data=array_total_featured_stage4_data

		scaler=MinMaxScaler()

		# Copy sample through GAN
		learning_rate = 0.0002

		n_hidden = 256
		n_input = stage_4_data.shape[1]
		n_noise = stage_4_data.shape[1]  


		X = tf.placeholder(tf.float32, [None, n_input])
		Z = tf.placeholder(tf.float32, [None, n_noise])

		G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden], stddev=0.01))
		G_b1 = tf.Variable(tf.zeros([n_hidden]))
		G_W2 = tf.Variable(tf.random_normal([n_hidden, n_input], stddev=0.01))
		G_b2 = tf.Variable(tf.zeros([n_input]))

		D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
		D_b1 = tf.Variable(tf.zeros([n_hidden]))
		D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01))
		D_b2 = tf.Variable(tf.zeros([1]))

		#generator 
		def generator(noise_z):
		
			hidden = tf.nn.relu(
					  tf.matmul(noise_z, G_W1) + G_b1)
		
		
		
			output = tf.nn.sigmoid(
					  tf.matmul(hidden, G_W2) + G_b2)

			return output

		#discriminator
		def discriminator(inputs):
		
			hidden = tf.nn.relu(
					  tf.matmul(inputs, D_W1) + D_b1)
		
		
		
			output = tf.nn.sigmoid(
						tf.matmul(hidden, D_W2) + D_b2)

			return output

		#Noise generation considering mean and variance
		def get_noise(batch_size, n_noise,mean, stddev):
			random_result=[]
			for i in range(batch_size*n_noise):
				tmp=np.random.randn()
				normalization_result=stddev*tmp+mean
				random_result.append(normalization_result)
			array_random_result=np.array(random_result)
			T_array_random_result=array_random_result.reshape(1,-1)
			return T_array_random_result


		G = generator(Z)
		D_gene = discriminator(G)

		D_real = discriminator(X)


		loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))

		loss_G = tf.reduce_mean(tf.log(D_gene))


		D_var_list = [D_W1, D_b1, D_W2, D_b2]
		G_var_list = [G_W1, G_b1, G_W2, G_b2]

		train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D,
													var_list=D_var_list)
		train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G,   
												var_list=G_var_list)


		one_1_hot=np.zeros((len(stage_1_samples),2))
		one_1_hot[np.arange(len(stage_1_samples))]=1

		one_2_hot=np.zeros((len(stage_2_samples),2))
		one_2_hot[np.arange(len(stage_2_samples))]=1

		one_3_hot=np.zeros((len(stage_3_samples),2))
		one_3_hot[np.arange(len(stage_3_samples))]=1

		one_4_hot=np.zeros((len(stage_4_samples),2))
		one_4_hot[np.arange(len(stage_4_samples))]=1


		stage_def_name = stage_1_data
		
		if stage_def == 1:
			stage_def_name = stage_1_data
		elif stage_def == 2:
			stage_def_name = stage_2_data
		elif stage_def == 3:
			stage_def_name = stage_3_data
		elif stage_def == 4:
			stage_def_name = stage_4_data
		print(str(stage_def) + " : " + str(stage_def_name.shape))
	
		data_normalization_result=scaler.fit_transform(stage_def_name)

		array_data_normalization_result=np.array(data_normalization_result)



		feature_names  =array_data_normalization_result.T
		print(feature_names.shape)

		feature_names =(feature_names).T

		feature_label=[]
		for i in range(feature_names.shape[0]):
			print(str(i) + "__" + str(feature_names.shape[0])+ "//" + str(feature_names.shape))
			feature_label.append(1)

		tmp_X_train,tmp_X_test,tmp_y_train,tmp_y_test=train_test_split(feature_names,feature_label,test_size=0.3,random_state=ith)
		X_train=tmp_X_train
		X_test=tmp_X_test
		#### save test data####
		np.save("1212blca/CASE_"+str(ith)+"_1014_TEST_Result"+"_"+sample_name+"_"+"stage"+str(stage_def),X_test)



		loss_val_D, loss_val_G=0,0

		sess=tf.Session()
		sess.run(tf.global_variables_initializer())

		correlation_result_total=[]
		D_real_score_total=[]
		D_gene_score_total=[]
		generator_score_total=[]

		Total_D_loss=[]
		Total_G_loss=[]
		Total_D_data=[]
		Total_G_data=[]
		Total_correlation_result=[]


		#Train
		Train_mean = np.mean(X_train,axis=1)
		Train_var = np.std(X_train,axis=1)
		train_num = len(X_train)-1
		epochs=1000
		sess.run(tf.global_variables_initializer())
		for epoch in range(epochs):

			for i in range(len(X_train)):
				one_X_train=X_train[i].reshape(1,-1)
				noise=get_noise(1, n_noise, np.mean(one_X_train) , np.std(one_X_train))
				_,loss_val_D=sess.run([train_D, loss_D],
							 feed_dict={X:one_X_train, Z:noise})


				_, loss_val_G=sess.run([train_G,loss_G],
							  feed_dict={Z:noise})

				D_real_score=sess.run(D_real,feed_dict={X:one_X_train,Z:noise})
				D_gene_score=sess.run(D_gene,feed_dict={X:one_X_train,Z:noise})




			Total_D_loss.append(loss_val_D)
			Total_G_loss.append(loss_val_G)

		############stage1 generation###############################
		if stage_number==0:
			Total_G_data=[]
			Total_correlation_result=[]
			#############input the number of data corresponding to GAN1 of stage1 in the range part#########################
			for m in range(2):
				choice = randint(0,train_num)
				tmp_mean = Train_mean[choice]
				tmp_var = Train_var[choice]
				noise=get_noise(1, n_noise, tmp_mean,tmp_var)
				generator_score=sess.run(G,feed_dict={Z:noise})
				correlation_result=np.corrcoef(generator_score,X_train[choice].reshape(1,-1))
				Total_correlation_result.append(correlation_result[0][1])   
				Total_G_data.append(generator_score)   


			sample_name = sample_name.split('_')[0]
			#### save the gerated GAN data####
			np.save("1212blca/CASE_"+str(ith)+"_1025_mutation_correlation_stage" + sample_name + "_" + str(stage_def)+"MUL1", Total_correlation_result)
			np.save("1212blca/CASE_"+str(ith)+"_1025_mutation_Generator_data_stage" + sample_name + "_" + str(stage_def)+"MUL1", Total_G_data)

			Total_correlation_result=[]
			Total_G_data=[]   
			#############input the number of data corresponding to GAN5 of stage1 in the range part#########################
			for m in range(10):
				choice = randint(0,train_num)
				tmp_mean = Train_mean[choice]
				tmp_var = Train_var[choice]
				noise=get_noise(1, n_noise, tmp_mean,tmp_var)
				generator_score=sess.run(G,feed_dict={Z:noise})
				correlation_result=np.corrcoef(generator_score,X_train[choice].reshape(1,-1))
				Total_correlation_result.append(correlation_result[0][1])   
				Total_G_data.append(generator_score)   


			sample_name = sample_name.split('_')[0]
			#### save the gerated GAN data####
			np.save("1212blca/CASE_"+str(ith)+"_1025_mutation_correlation_stage" + sample_name + "_" + str(stage_def)+"MUL5", Total_correlation_result)
			np.save("1212blca/CASE_"+str(ith)+"_1025_mutation_Generator_data_stage" + sample_name + "_" +str(stage_def)+"MUL5", Total_G_data)  



			Total_correlation_result=[]
			Total_G_data=[] 
			#############input the number of data corresponding to GAN20 of stage1 in the range part#########################
			for m in range(40):
				choice = randint(0,train_num)
				tmp_mean = Train_mean[choice]
				tmp_var = Train_var[choice]
				noise=get_noise(1, n_noise, tmp_mean,tmp_var)
				generator_score=sess.run(G,feed_dict={Z:noise})
				correlation_result=np.corrcoef(generator_score,X_train[choice].reshape(1,-1))
				Total_correlation_result.append(correlation_result[0][1])   
				Total_G_data.append(generator_score)   


			sample_name = sample_name.split('_')[0]
			#### save the gerated GAN data####
			np.save("1212blca/CASE_"+str(ith)+"_1025_mutation_correlation_stage" + sample_name + "_" + str(stage_def)+"MUL20", Total_correlation_result)
			np.save("1212blca/CASE_"+str(ith)+"_1025_mutation_Generator_data_stage" + sample_name + "_" + str(stage_def)+"MUL20", Total_G_data)  

			Total_correlation_result=[]
			Total_G_data=[] 
			#############input the number of data corresponding to GAN100 of stage1 in the range part#########################
			for m in range(200):
				choice = randint(0,train_num)
				tmp_mean = Train_mean[choice]
				tmp_var = Train_var[choice]
				noise=get_noise(1, n_noise, tmp_mean,tmp_var)
				generator_score=sess.run(G,feed_dict={Z:noise})
				correlation_result=np.corrcoef(generator_score,X_train[choice].reshape(1,-1))
				Total_correlation_result.append(correlation_result[0][1])   
				Total_G_data.append(generator_score)   


			sample_name = sample_name.split('_')[0]
			#### save the gerated GAN data####
			np.save("1212blca/CASE_"+str(ith)+"_1025_mutation_correlation_stage" + sample_name + "_" + str(stage_def)+"MUL100", Total_correlation_result)
			np.save("1212blca/CASE_"+str(ith)+"_1025_mutation_Generator_data_stage" + sample_name + "_" + str(stage_def)+"MUL100", Total_G_data)  
		############stage2 generation###############################
		if stage_number==1:		
			Total_G_data=[]
			Total_correlation_result=[]
			#############input the number of data corresponding to GAN1 of stage2 in the range part#########################
			for m in range(79):
				choice = randint(0,train_num)
				tmp_mean = Train_mean[choice]
				tmp_var = Train_var[choice]
				noise=get_noise(1, n_noise, tmp_mean,tmp_var)
				generator_score=sess.run(G,feed_dict={Z:noise})
				correlation_result=np.corrcoef(generator_score,X_train[choice].reshape(1,-1))
				Total_correlation_result.append(correlation_result[0][1])   
				Total_G_data.append(generator_score)   


			sample_name = sample_name.split('_')[0]
			#### save the gerated GAN data####
			np.save("1212blca/CASE_"+str(ith)+"_1025_mutation_correlation_stage" + sample_name + "_" + str(stage_def)+"MUL1", Total_correlation_result)
			np.save("1212blca/CASE_"+str(ith)+"_1025_mutation_Generator_data_stage" + sample_name + "_" + str(stage_def)+"MUL1", Total_G_data)

			Total_correlation_result=[]
			Total_G_data=[]   
			#############input the number of data corresponding to GAN5 of stage2 in the range part#########################
			for m in range(395):
				choice = randint(0,train_num)
				tmp_mean = Train_mean[choice]
				tmp_var = Train_var[choice]
				noise=get_noise(1, n_noise, tmp_mean,tmp_var)
				generator_score=sess.run(G,feed_dict={Z:noise})
				correlation_result=np.corrcoef(generator_score,X_train[choice].reshape(1,-1))
				Total_correlation_result.append(correlation_result[0][1])   
				Total_G_data.append(generator_score)   


			sample_name = sample_name.split('_')[0]
			#### save the gerated GAN data####
			np.save("1212blca/CASE_"+str(ith)+"_1025_mutation_correlation_stage" + sample_name + "_" + str(stage_def)+"MUL5", Total_correlation_result)
			np.save("1212blca/CASE_"+str(ith)+"_1025_mutation_Generator_data_stage" + sample_name + "_" + str(stage_def)+"MUL5", Total_G_data)  
		   

		 
			Total_correlation_result=[]
			Total_G_data=[] 
			#############input the number of data corresponding to GAN20 of stage2 in the range part.#########################
			for m in range(1580):
				choice = randint(0,train_num)
				tmp_mean = Train_mean[choice]
				tmp_var = Train_var[choice]
				noise=get_noise(1, n_noise, tmp_mean,tmp_var)
				generator_score=sess.run(G,feed_dict={Z:noise})
				correlation_result=np.corrcoef(generator_score,X_train[choice].reshape(1,-1))
				Total_correlation_result.append(correlation_result[0][1])   
				Total_G_data.append(generator_score)   


			sample_name = sample_name.split('_')[0]
			#### save the gerated GAN data####
			np.save("1212blca/CASE_"+str(ith)+"_1025_mutation_correlation_stage" + sample_name + "_" + str(stage_def)+"MUL20", Total_correlation_result)
			np.save("1212blca/CASE_"+str(ith)+"_1025_mutation_Generator_data_stage" + sample_name + "_" + str(stage_def)+"MUL20", Total_G_data)  
		   
			Total_correlation_result=[]
			Total_G_data=[] 
			#############input the number of data corresponding to GAN100 of stage2 in the range part#########################
			for m in range(7900):
				choice = randint(0,train_num)
				tmp_mean = Train_mean[choice]
				tmp_var = Train_var[choice]
				noise=get_noise(1, n_noise, tmp_mean,tmp_var)
				generator_score=sess.run(G,feed_dict={Z:noise})
				correlation_result=np.corrcoef(generator_score,X_train[choice].reshape(1,-1))
				Total_correlation_result.append(correlation_result[0][1])   
				Total_G_data.append(generator_score)   


			sample_name = sample_name.split('_')[0]
			#### save the gerated GAN data####
			np.save("1212blca/CASE_"+str(ith)+"_1025_mutation_correlation_stage" + sample_name + "_" + str(stage_def)+"MUL100", Total_correlation_result)
			np.save("1212blca/CASE_"+str(ith)+"_1025_mutation_Generator_data_stage" + sample_name + "_" + str(stage_def)+"MUL100", Total_G_data)  
		
		############stage3 generation###############################
		if stage_number==2:
			#############input the number of data corresponding to GAN1 of stage3 in the range part#########################
			for m in range(87):
				choice = randint(0,train_num)
				tmp_mean = Train_mean[choice]
				tmp_var = Train_var[choice]
				noise=get_noise(1, n_noise, tmp_mean,tmp_var)
				generator_score=sess.run(G,feed_dict={Z:noise})
				correlation_result=np.corrcoef(generator_score,X_train[choice].reshape(1,-1))
				Total_correlation_result.append(correlation_result[0][1])   
				Total_G_data.append(generator_score)   


			sample_name = sample_name.split('_')[0]
			#### save the gerated GAN data####
			np.save("1212blca/CASE_"+str(ith)+"_1025_mutation_correlation_stage" + sample_name + "_" + str(stage_def)+"MUL1", Total_correlation_result)
			np.save("1212blca/CASE_"+str(ith)+"_1025_mutation_Generator_data_stage" + sample_name + "_" + str(stage_def)+"MUL1", Total_G_data)

			Total_correlation_result=[]
			Total_G_data=[]   
			#############input the number of data corresponding to GAN5 of stage3 in the range part#########################
			for m in range(435):
				choice = randint(0,train_num)
				tmp_mean = Train_mean[choice]
				tmp_var = Train_var[choice]
				noise=get_noise(1, n_noise, tmp_mean,tmp_var)
				generator_score=sess.run(G,feed_dict={Z:noise})
				correlation_result=np.corrcoef(generator_score,X_train[choice].reshape(1,-1))
				Total_correlation_result.append(correlation_result[0][1])   
				Total_G_data.append(generator_score)   


			sample_name = sample_name.split('_')[0]
			#### save the gerated GAN data####
			np.save("1212blca/CASE_"+str(ith)+"_1025_mutation_correlation_stage" + sample_name + "_" + str(stage_def)+"MUL5", Total_correlation_result)
			np.save("1212blca/CASE_"+str(ith)+"_1025_mutation_Generator_data_stage" + sample_name + "_" + str(stage_def)+"MUL5", Total_G_data)  
		   


			Total_correlation_result=[]
			Total_G_data=[] 
			#############input the number of data corresponding to GAN20 of stage3 in the range part#########################
			for m in range(1740):
				choice = randint(0,train_num)
				tmp_mean = Train_mean[choice]
				tmp_var = Train_var[choice]
				noise=get_noise(1, n_noise, tmp_mean,tmp_var)
				generator_score=sess.run(G,feed_dict={Z:noise})
				correlation_result=np.corrcoef(generator_score,X_train[choice].reshape(1,-1))
				Total_correlation_result.append(correlation_result[0][1])   
				Total_G_data.append(generator_score)   


			sample_name = sample_name.split('_')[0]

			np.save("1212blca/CASE_"+str(ith)+"_1025_mutation_correlation_stage" + sample_name + "_" + str(stage_def)+"MUL20", Total_correlation_result)
			np.save("1212blca/CASE_"+str(ith)+"_1025_mutation_Generator_data_stage" + sample_name + "_" + str(stage_def)+"MUL20", Total_G_data)  
			#############input the number of data corresponding to GAN100 of stage3 in the range part########################
			Total_correlation_result=[]
			Total_G_data=[] 
			for m in range(8700):
				choice = randint(0,train_num)
				tmp_mean = Train_mean[choice]
				tmp_var = Train_var[choice]
				noise=get_noise(1, n_noise, tmp_mean,tmp_var)
				generator_score=sess.run(G,feed_dict={Z:noise})
				correlation_result=np.corrcoef(generator_score,X_train[choice].reshape(1,-1))
				Total_correlation_result.append(correlation_result[0][1])   
				Total_G_data.append(generator_score)   


			sample_name = sample_name.split('_')[0]
			#### save the gerated GAN data####
			np.save("1212blca/CASE_"+str(ith)+"_1025_mutation_correlation_stage" + sample_name + "_" + str(stage_def)+"MUL100", Total_correlation_result)
			np.save("1212blca/CASE_"+str(ith)+"_1025_mutation_Generator_data_stage" + sample_name + "_" + str(stage_def)+"MUL100", Total_G_data)  
		############stage4 generation###############################			
		if stage_number==3:
			Total_correlation_result=[]
			Total_G_data=[] 
			#############input the number of data corresponding to GAN1 of stage4 in the range part#########################			
			for m in range(83):
				choice = randint(0,train_num)
				tmp_mean = Train_mean[choice]
				tmp_var = Train_var[choice]
				noise=get_noise(1, n_noise, tmp_mean,tmp_var)
				generator_score=sess.run(G,feed_dict={Z:noise})
				correlation_result=np.corrcoef(generator_score,X_train[choice].reshape(1,-1))
				Total_correlation_result.append(correlation_result[0][1])   
				Total_G_data.append(generator_score)   


			sample_name = sample_name.split('_')[0]
			#### save the gerated GAN data####
			np.save("1212blca/CASE_"+str(ith)+"_1025_mutation_correlation_stage" + sample_name + "_" + str(stage_def)+"MUL1", Total_correlation_result)
			np.save("1212blca/CASE_"+str(ith)+"_1025_mutation_Generator_data_stage" + sample_name + "_" + str(stage_def)+"MUL1", Total_G_data)

			Total_correlation_result=[]
			Total_G_data=[]   
			#############input the number of data corresponding to GAN5 of stage4 in the range part#########################
			for m in range(415):
				choice = randint(0,train_num)
				tmp_mean = Train_mean[choice]
				tmp_var = Train_var[choice]
				noise=get_noise(1, n_noise, tmp_mean,tmp_var)
				generator_score=sess.run(G,feed_dict={Z:noise})
				correlation_result=np.corrcoef(generator_score,X_train[choice].reshape(1,-1))
				Total_correlation_result.append(correlation_result[0][1])   
				Total_G_data.append(generator_score)   


			sample_name = sample_name.split('_')[0]
			#### save the gerated GAN data####
			np.save("1212blca/CASE_"+str(ith)+"_1025_mutation_correlation_stage" + sample_name + "_" + str(stage_def)+"MUL5", Total_correlation_result)
			np.save("1212blca/CASE_"+str(ith)+"_1025_mutation_Generator_data_stage" + sample_name + "_" + str(stage_def)+"MUL5", Total_G_data)  



			Total_correlation_result=[]
			Total_G_data=[] 
			#############input the number of data corresponding to GAN20 of stage4 in the range part#########################
			for m in range(1660):
				choice = randint(0,train_num)
				tmp_mean = Train_mean[choice]
				tmp_var = Train_var[choice]
				noise=get_noise(1, n_noise, tmp_mean,tmp_var)
				generator_score=sess.run(G,feed_dict={Z:noise})
				correlation_result=np.corrcoef(generator_score,X_train[choice].reshape(1,-1))
				Total_correlation_result.append(correlation_result[0][1])   
				Total_G_data.append(generator_score)   


			sample_name = sample_name.split('_')[0]
			#### save the gerated GAN data####
			np.save("1212blca/CASE_"+str(ith)+"_1025_mutation_correlation_stage" + sample_name + "_" + str(stage_def)+"MUL20", Total_correlation_result)
			np.save("1212blca/CASE_"+str(ith)+"_1025_mutation_Generator_data_stage" + sample_name + "_" + str(stage_def)+"MUL20", Total_G_data)  

			Total_correlation_result=[]
			Total_G_data=[] 
			#############input the number of data corresponding to GAN100 of stage4 in the range part#########################
			for m in range(8300):
				choice = randint(0,train_num)
				tmp_mean = Train_mean[choice]
				tmp_var = Train_var[choice]
				noise=get_noise(1, n_noise, tmp_mean,tmp_var)
				generator_score=sess.run(G,feed_dict={Z:noise})
				correlation_result=np.corrcoef(generator_score,X_train[choice].reshape(1,-1))
				Total_correlation_result.append(correlation_result[0][1])   
				Total_G_data.append(generator_score)   


			sample_name = sample_name.split('_')[0]
			#### save the gerated GAN data####
			np.save("1212blca/CASE_"+str(ith)+"_1025_mutation_correlation_stage" + sample_name + "_" + str(stage_def)+"MUL100", Total_correlation_result)
			np.save("1212blca/CASE_"+str(ith)+"_1025_mutation_Generator_data_stage" + sample_name + "_" + str(stage_def)+"MUL100", Total_G_data)  
			
			
############################# accuracy evaluation part ##########################################

for iter in range(4):
	if iter==0:
		mul=1
		f=open("/Data3/chkwon/Data3_Storage/1212_Data/blca_MUL"+str(mul)+"txt",'w')
		for j in range(30):
		################################## GAN 1 ######################################	
			###########Loading GAN1 data corresponding to stage1################################
			Total_correlation_result=np.load('/Data3/chkwon/Data3_Storage/1212blca/CASE_'+str(j)+'_1025_mutation_correlation_stageblca.txt_1MUL'+str(mul)+'.npy')
			Total_G_data=np.load("/Data3/chkwon/Data3_Storage/1212blca/CASE_"+str(j)+"_1025_mutation_Generator_data_stageblca.txt_1MUL"+str(mul)+".npy")
			new_correlation1_result=[]
			line_tmp=[]
			for i in range(len(Total_correlation_result)):
				tmp=Total_correlation_result[i]
				line_tmp.append(i)
				line_tmp.append(tmp)
				new_correlation1_result.append(line_tmp)
				line_tmp=[]
			new_correlation1_result.sort(key=itemgetter(1),reverse=True)

			###########Loading GAN1 data corresponding to stage2################################
			Total_correlation2_result=np.load('/Data3/chkwon/Data3_Storage/1212blca/CASE_'+str(j)+'_1025_mutation_correlation_stageblca.txt_2MUL'+str(mul)+'.npy')
			Total_G2_data=np.load("/Data3/chkwon/Data3_Storage/1212blca/CASE_"+str(j)+"_1025_mutation_Generator_data_stageblca.txt_2MUL"+str(mul)+".npy")
			new_correlation2_result=[]
			line_tmp=[]
			for i in range(len(Total_correlation2_result)):
				tmp=Total_correlation2_result[i]
				line_tmp.append(i)
				line_tmp.append(tmp)
				new_correlation2_result.append(line_tmp)
				line_tmp=[]
			new_correlation2_result.sort(key=itemgetter(1),reverse=True)

			###########Loading GAN1 data corresponding to stage3################################
			Total_correlation3_result=np.load('/Data3/chkwon/Data3_Storage/1212blca/CASE_'+str(j)+'_1025_mutation_correlation_stageblca.txt_3MUL'+str(mul)+'.npy')
			Total_G3_data=np.load("/Data3/chkwon/Data3_Storage/1212blca/CASE_"+str(j)+"_1025_mutation_Generator_data_stageblca.txt_3MUL"+str(mul)+".npy")
			new_correlation3_result=[]
			line_tmp=[]
			for i in range(len(Total_correlation3_result)):
				tmp=Total_correlation3_result[i]
				line_tmp.append(i)
				line_tmp.append(tmp)
				new_correlation3_result.append(line_tmp)
				line_tmp=[]
			new_correlation3_result.sort(key=itemgetter(1),reverse=True)
			###########Loading GAN1 data corresponding to stage4################################
			Total_correlation4_result=np.load('/Data3/chkwon/Data3_Storage/1212blca/CASE_'+str(j)+'_1025_mutation_correlation_stageblca.txt_4MUL'+str(mul)+'.npy')
			Total_G4_data=np.load("/Data3/chkwon/Data3_Storage/1212blca//CASE_"+str(j)+"_1025_mutation_Generator_data_stageblca.txt_4MUL"+str(mul)+".npy")
			new_correlation4_result=[]
			line_tmp=[]
			for i in range(len(Total_correlation4_result)):
				tmp=Total_correlation4_result[i]
				line_tmp.append(i)
				line_tmp.append(tmp)
				new_correlation4_result.append(line_tmp)
				line_tmp=[]
			new_correlation4_result.sort(key=itemgetter(1),reverse=True)
			###########Enter the number of data corresponding to GAN1 for each stage in the range################################
			new_correlation4_result_index=[]
			for i in range(83):
				tmp=new_correlation4_result[i][0]
				new_correlation4_result_index.append(tmp)
				
			new_correlation3_result_index=[]
			for i in range(87):
				tmp=new_correlation3_result[i][0]
				new_correlation3_result_index.append(tmp)
				
			new_correlation2_result_index=[]
			for i in range(79):
				tmp=new_correlation2_result[i][0]
				new_correlation2_result_index.append(tmp)

			new_correlation1_result_index=[]
			for i in range(2):
				tmp=new_correlation1_result[i][0]
				new_correlation1_result_index.append(tmp)

			sample_Total_G4_data=[]
			for i in range(Total_G4_data.shape[0]):
				tmp=Total_G4_data[i][0]
				sample_Total_G4_data.append(tmp)
			sample_Total_G4_data=np.array(sample_Total_G4_data)


			sample_Total_G3_data=[]
			for i in range(Total_G3_data.shape[0]):
				tmp=Total_G3_data[i][0]
				sample_Total_G3_data.append(tmp)
				
			sample_Total_G3_data=np.array(sample_Total_G3_data)


			sample_Total_G2_data=[]
			for i in range(Total_G2_data.shape[0]):
				tmp=Total_G2_data[i][0]
				sample_Total_G2_data.append(tmp)
				
			sample_Total_G2_data=np.array(sample_Total_G2_data)

			sample_Total_G_data=[]
			for i in range(Total_G_data.shape[0]):
				tmp=Total_G_data[i][0]
				sample_Total_G_data.append(tmp)
				
			sample_Total_G_data=np.array(sample_Total_G_data)


			Train_4_data=[]
			for i in range(len(new_correlation4_result_index)):
				tmp=sample_Total_G4_data[new_correlation4_result_index[i]]
				Train_4_data.append(tmp)
			Train_4_data=np.array(Train_4_data)


			Train_3_data=[]
			for i in range(len(new_correlation3_result_index)):
				tmp=sample_Total_G3_data[new_correlation3_result_index[i]]
				Train_3_data.append(tmp)
			Train_3_data=np.array(Train_3_data)


			Train_2_data=[]
			for i in range(len(new_correlation2_result_index)):
				tmp=sample_Total_G2_data[new_correlation2_result_index[i]]
				Train_2_data.append(tmp)
			Train_2_data=np.array(Train_2_data)

			Train_1_data=[]
			for i in range(len(new_correlation1_result_index)):
				tmp=sample_Total_G_data[new_correlation1_result_index[i]]
				Train_1_data.append(tmp)
			Train_1_data=np.array(Train_1_data)

			stage_1_2_data=np.vstack((Train_4_data,Train_3_data))

			stage_1_2_3_data=np.vstack((stage_1_2_data,Train_2_data))

			stage_1_2_3_4_data=np.vstack((stage_1_2_3_data,Train_1_data))

			###########Enter the number of data corresponding to GAN1 for each stage in the range################################
			new_train_label=[]
			for i in range(83):
				new_train_label.append(4)
			for i in range(87):
				new_train_label.append(3)
			for i in range(79):
				new_train_label.append(2)
			for i in range(2):
				new_train_label.append(1)

			###########Load test data################################
			newnew_abnormal_data1=np.load("/Data3/chkwon/Data3_Storage/1212blca/CASE_"+str(j)+"_1014_TEST_Result_blca.txt_stage1.npy")
			newnew_abnormal_data2=np.load("/Data3/chkwon/Data3_Storage/1212blca//CASE_"+str(j)+"_1014_TEST_Result_blca.txt_stage2.npy")
			newnew_abnormal_data3=np.load("/Data3/chkwon/Data3_Storage/1212blca/CASE_"+str(j)+"_1014_TEST_Result_blca.txt_stage3.npy")
			newnew_abnormal_data4=np.load("/Data3/chkwon/Data3_Storage/1212blca/CASE_"+str(j)+"_1014_TEST_Result_blca.txt_stage4.npy")


			newnew_abnormal_data12=np.vstack((newnew_abnormal_data1,newnew_abnormal_data2))
			newnew_abnormal_data123=np.vstack((newnew_abnormal_data12,newnew_abnormal_data3))
			newnew_abnormal_data1234=np.vstack((newnew_abnormal_data123,newnew_abnormal_data4))

			test_label=[]
			for i in range(newnew_abnormal_data1.shape[0]):
				test_label.append(1)
			for i in range(newnew_abnormal_data2.shape[0]):
				test_label.append(2)
			for i in range(newnew_abnormal_data3.shape[0]):
				test_label.append(3)
			for i in range(newnew_abnormal_data4.shape[0]):
				test_label.append(4)
				
			X_train=stage_1_2_3_4_data
	
			X_train=pd.DataFrame(data=X_train)
			
			
			Y_train=new_train_label
			Y_train=np.array(Y_train)
		
			Y_train=to_categorical(Y_train,num_classes=4)
			
			
			X_test=newnew_abnormal_data1234
				
			X_test=pd.DataFrame(data=X_test)

			print("test_label",test_label)
			
			print("test_label",test_label)
			Y_test=test_label
			Y_test=np.array(Y_test)
			Y_test=to_categorical(Y_test,num_classes=4)
			
			print("X_train.shape",X_train.shape)
			print("Y_train.shape",Y_train.shape)
			
			accuracy=CNN1D_evaluate_model(X_train,X_test,Y_train,Y_test,Y_test)
			print("all_score",accuracy)
			print("all_score[0]",accuracy[0])
			print("all_score[1]",accuracy[1])
			print("all_score[2]",accuracy[2])
			print("all_score_length",len(accuracy))
			print("all_score_type",type(accuracy))
			#data="%f \n"%accuracy
			f.write("accuracy")
			f.write(str(accuracy[0]))
			f.write("\n")
			f.write(str(accuracy[1]))
		f.close()
	if iter==1:
		mul=5
		f=open("/Data3/chkwon/Data3_Storage/1212_Data/blca_MUL"+str(mul)+"txt",'w')
		for j in range(30):
		################################## GAN 5 ######################################	
			###########Loading GAN5 data corresponding to stage1################################
			Total_correlation_result=np.load('/Data3/chkwon/Data3_Storage/1212blca/CASE_'+str(j)+'_1025_mutation_correlation_stageblca.txt_1MUL'+str(mul)+'.npy')
			Total_G_data=np.load("/Data3/chkwon/Data3_Storage/1212blca/CASE_"+str(j)+"_1025_mutation_Generator_data_stageblca.txt_1MUL"+str(mul)+".npy")
			new_correlation1_result=[]
			line_tmp=[]
			for i in range(len(Total_correlation_result)):
				tmp=Total_correlation_result[i]
				line_tmp.append(i)
				line_tmp.append(tmp)
				new_correlation1_result.append(line_tmp)
				line_tmp=[]
			new_correlation1_result.sort(key=itemgetter(1),reverse=True)

			###########Loading GAN5 data corresponding to stage2################################
			Total_correlation2_result=np.load('/Data3/chkwon/Data3_Storage/1212blca/CASE_'+str(j)+'_1025_mutation_correlation_stageblca.txt_2MUL'+str(mul)+'.npy')
			Total_G2_data=np.load("/Data3/chkwon/Data3_Storage/1212blca/CASE_"+str(j)+"_1025_mutation_Generator_data_stageblca.txt_2MUL"+str(mul)+".npy")
			new_correlation2_result=[]
			line_tmp=[]
			for i in range(len(Total_correlation2_result)):
				tmp=Total_correlation2_result[i]
				line_tmp.append(i)
				line_tmp.append(tmp)
				new_correlation2_result.append(line_tmp)
				line_tmp=[]
			new_correlation2_result.sort(key=itemgetter(1),reverse=True)

			###########Loading GAN5 data corresponding to stage3################################
			Total_correlation3_result=np.load('/Data3/chkwon/Data3_Storage/1212blca/CASE_'+str(j)+'_1025_mutation_correlation_stageblca.txt_3MUL'+str(mul)+'.npy')
			Total_G3_data=np.load("/Data3/chkwon/Data3_Storage/1212blca/CASE_"+str(j)+"_1025_mutation_Generator_data_stageblca.txt_3MUL"+str(mul)+".npy")
			new_correlation3_result=[]
			line_tmp=[]
			for i in range(len(Total_correlation3_result)):
				tmp=Total_correlation3_result[i]
				line_tmp.append(i)
				line_tmp.append(tmp)
				new_correlation3_result.append(line_tmp)
				line_tmp=[]
			new_correlation3_result.sort(key=itemgetter(1),reverse=True)
			###########Loading GAN5 data corresponding to stage4################################
			Total_correlation4_result=np.load('/Data3/chkwon/Data3_Storage/1212blca/CASE_'+str(j)+'_1025_mutation_correlation_stageblca.txt_4MUL'+str(mul)+'.npy')
			Total_G4_data=np.load("/Data3/chkwon/Data3_Storage/1212blca//CASE_"+str(j)+"_1025_mutation_Generator_data_stageblca.txt_4MUL"+str(mul)+".npy")
			new_correlation4_result=[]
			line_tmp=[]
			for i in range(len(Total_correlation4_result)):
				tmp=Total_correlation4_result[i]
				line_tmp.append(i)
				line_tmp.append(tmp)
				new_correlation4_result.append(line_tmp)
				line_tmp=[]
			new_correlation4_result.sort(key=itemgetter(1),reverse=True)
			###########Enter the number of data corresponding to GAN5 for each stage in the range################################
			new_correlation4_result_index=[]
			for i in range(83):
				tmp=new_correlation4_result[i][0]
				new_correlation4_result_index.append(tmp)
				
			new_correlation3_result_index=[]
			for i in range(87):
				tmp=new_correlation3_result[i][0]
				new_correlation3_result_index.append(tmp)
				
			new_correlation2_result_index=[]
			for i in range(79):
				tmp=new_correlation2_result[i][0]
				new_correlation2_result_index.append(tmp)

			new_correlation1_result_index=[]
			for i in range(2):
				tmp=new_correlation1_result[i][0]
				new_correlation1_result_index.append(tmp)

			sample_Total_G4_data=[]
			for i in range(Total_G4_data.shape[0]):
				tmp=Total_G4_data[i][0]
				sample_Total_G4_data.append(tmp)
			sample_Total_G4_data=np.array(sample_Total_G4_data)


			sample_Total_G3_data=[]
			for i in range(Total_G3_data.shape[0]):
				tmp=Total_G3_data[i][0]
				sample_Total_G3_data.append(tmp)
				
			sample_Total_G3_data=np.array(sample_Total_G3_data)


			sample_Total_G2_data=[]
			for i in range(Total_G2_data.shape[0]):
				tmp=Total_G2_data[i][0]
				sample_Total_G2_data.append(tmp)
				
			sample_Total_G2_data=np.array(sample_Total_G2_data)

			sample_Total_G_data=[]
			for i in range(Total_G_data.shape[0]):
				tmp=Total_G_data[i][0]
				sample_Total_G_data.append(tmp)
				
			sample_Total_G_data=np.array(sample_Total_G_data)


			Train_4_data=[]
			for i in range(len(new_correlation4_result_index)):
				tmp=sample_Total_G4_data[new_correlation4_result_index[i]]
				Train_4_data.append(tmp)
			Train_4_data=np.array(Train_4_data)


			Train_3_data=[]
			for i in range(len(new_correlation3_result_index)):
				tmp=sample_Total_G3_data[new_correlation3_result_index[i]]
				Train_3_data.append(tmp)
			Train_3_data=np.array(Train_3_data)


			Train_2_data=[]
			for i in range(len(new_correlation2_result_index)):
				tmp=sample_Total_G2_data[new_correlation2_result_index[i]]
				Train_2_data.append(tmp)
			Train_2_data=np.array(Train_2_data)

			Train_1_data=[]
			for i in range(len(new_correlation1_result_index)):
				tmp=sample_Total_G_data[new_correlation1_result_index[i]]
				Train_1_data.append(tmp)
			Train_1_data=np.array(Train_1_data)

			stage_1_2_data=np.vstack((Train_4_data,Train_3_data))

			stage_1_2_3_data=np.vstack((stage_1_2_data,Train_2_data))

			stage_1_2_3_4_data=np.vstack((stage_1_2_3_data,Train_1_data))

			###########Enter the number of data corresponding to GAN5 for each stage in the range################################
			new_train_label=[]
			for i in range(83):
				new_train_label.append(4)
			for i in range(87):
				new_train_label.append(3)
			for i in range(79):
				new_train_label.append(2)
			for i in range(2):
				new_train_label.append(1)

			###########Load test data################################
			newnew_abnormal_data1=np.load("/Data3/chkwon/Data3_Storage/1212blca/CASE_"+str(j)+"_1014_TEST_Result_blca.txt_stage1.npy")
			newnew_abnormal_data2=np.load("/Data3/chkwon/Data3_Storage/1212blca//CASE_"+str(j)+"_1014_TEST_Result_blca.txt_stage2.npy")
			newnew_abnormal_data3=np.load("/Data3/chkwon/Data3_Storage/1212blca/CASE_"+str(j)+"_1014_TEST_Result_blca.txt_stage3.npy")
			newnew_abnormal_data4=np.load("/Data3/chkwon/Data3_Storage/1212blca/CASE_"+str(j)+"_1014_TEST_Result_blca.txt_stage4.npy")


			newnew_abnormal_data12=np.vstack((newnew_abnormal_data1,newnew_abnormal_data2))
			newnew_abnormal_data123=np.vstack((newnew_abnormal_data12,newnew_abnormal_data3))
			newnew_abnormal_data1234=np.vstack((newnew_abnormal_data123,newnew_abnormal_data4))

			test_label=[]
			for i in range(newnew_abnormal_data1.shape[0]):
				test_label.append(1)
			for i in range(newnew_abnormal_data2.shape[0]):
				test_label.append(2)
			for i in range(newnew_abnormal_data3.shape[0]):
				test_label.append(3)
			for i in range(newnew_abnormal_data4.shape[0]):
				test_label.append(4)
				
			X_train=stage_1_2_3_4_data
	
			X_train=pd.DataFrame(data=X_train)
			
			
			Y_train=new_train_label
			Y_train=np.array(Y_train)
		
			Y_train=to_categorical(Y_train,num_classes=4)
			
			
			X_test=newnew_abnormal_data1234
				
			X_test=pd.DataFrame(data=X_test)

			print("test_label",test_label)
			
			print("test_label",test_label)
			Y_test=test_label
			Y_test=np.array(Y_test)
			Y_test=to_categorical(Y_test,num_classes=4)
			
			print("X_train.shape",X_train.shape)
			print("Y_train.shape",Y_train.shape)
			
			accuracy=CNN1D_evaluate_model(X_train,X_test,Y_train,Y_test,Y_test)
			print("all_score",accuracy)
			print("all_score[0]",accuracy[0])
			print("all_score[1]",accuracy[1])
			print("all_score[2]",accuracy[2])
			print("all_score_length",len(accuracy))
			print("all_score_type",type(accuracy))
			#data="%f \n"%accuracy
			f.write("accuracy")
			f.write(str(accuracy[0]))
			f.write("\n")
			f.write(str(accuracy[1]))
		f.close()
		
	if iter==2:
		mul=20
		f=open("/Data3/chkwon/Data3_Storage/1212_Data/blca_MUL"+str(mul)+"txt",'w')
		for j in range(30):
		################################## GAN 20 ######################################	
			###########Loading GAN20 data corresponding to stage1################################
			Total_correlation_result=np.load('/Data3/chkwon/Data3_Storage/1212blca/CASE_'+str(j)+'_1025_mutation_correlation_stageblca.txt_1MUL'+str(mul)+'.npy')
			Total_G_data=np.load("/Data3/chkwon/Data3_Storage/1212blca/CASE_"+str(j)+"_1025_mutation_Generator_data_stageblca.txt_1MUL"+str(mul)+".npy")
			new_correlation1_result=[]
			line_tmp=[]
			for i in range(len(Total_correlation_result)):
				tmp=Total_correlation_result[i]
				line_tmp.append(i)
				line_tmp.append(tmp)
				new_correlation1_result.append(line_tmp)
				line_tmp=[]
			new_correlation1_result.sort(key=itemgetter(1),reverse=True)

			###########Loading GAN20 data corresponding to stage2################################
			Total_correlation2_result=np.load('/Data3/chkwon/Data3_Storage/1212blca/CASE_'+str(j)+'_1025_mutation_correlation_stageblca.txt_2MUL'+str(mul)+'.npy')
			Total_G2_data=np.load("/Data3/chkwon/Data3_Storage/1212blca/CASE_"+str(j)+"_1025_mutation_Generator_data_stageblca.txt_2MUL"+str(mul)+".npy")
			new_correlation2_result=[]
			line_tmp=[]
			for i in range(len(Total_correlation2_result)):
				tmp=Total_correlation2_result[i]
				line_tmp.append(i)
				line_tmp.append(tmp)
				new_correlation2_result.append(line_tmp)
				line_tmp=[]
			new_correlation2_result.sort(key=itemgetter(1),reverse=True)

			###########Loading GAN20 data corresponding to stage3################################
			Total_correlation3_result=np.load('/Data3/chkwon/Data3_Storage/1212blca/CASE_'+str(j)+'_1025_mutation_correlation_stageblca.txt_3MUL'+str(mul)+'.npy')
			Total_G3_data=np.load("/Data3/chkwon/Data3_Storage/1212blca/CASE_"+str(j)+"_1025_mutation_Generator_data_stageblca.txt_3MUL"+str(mul)+".npy")
			new_correlation3_result=[]
			line_tmp=[]
			for i in range(len(Total_correlation3_result)):
				tmp=Total_correlation3_result[i]
				line_tmp.append(i)
				line_tmp.append(tmp)
				new_correlation3_result.append(line_tmp)
				line_tmp=[]
			new_correlation3_result.sort(key=itemgetter(1),reverse=True)
			###########Loading GAN20 data corresponding to stage4################################
			Total_correlation4_result=np.load('/Data3/chkwon/Data3_Storage/1212blca/CASE_'+str(j)+'_1025_mutation_correlation_stageblca.txt_4MUL'+str(mul)+'.npy')
			Total_G4_data=np.load("/Data3/chkwon/Data3_Storage/1212blca//CASE_"+str(j)+"_1025_mutation_Generator_data_stageblca.txt_4MUL"+str(mul)+".npy")
			new_correlation4_result=[]
			line_tmp=[]
			for i in range(len(Total_correlation4_result)):
				tmp=Total_correlation4_result[i]
				line_tmp.append(i)
				line_tmp.append(tmp)
				new_correlation4_result.append(line_tmp)
				line_tmp=[]
			new_correlation4_result.sort(key=itemgetter(1),reverse=True)
			###########Enter the number of data corresponding to GAN20 for each stage in the range################################
			new_correlation4_result_index=[]
			for i in range(83):
				tmp=new_correlation4_result[i][0]
				new_correlation4_result_index.append(tmp)
				
			new_correlation3_result_index=[]
			for i in range(87):
				tmp=new_correlation3_result[i][0]
				new_correlation3_result_index.append(tmp)
				
			new_correlation2_result_index=[]
			for i in range(79):
				tmp=new_correlation2_result[i][0]
				new_correlation2_result_index.append(tmp)

			new_correlation1_result_index=[]
			for i in range(2):
				tmp=new_correlation1_result[i][0]
				new_correlation1_result_index.append(tmp)

			sample_Total_G4_data=[]
			for i in range(Total_G4_data.shape[0]):
				tmp=Total_G4_data[i][0]
				sample_Total_G4_data.append(tmp)
			sample_Total_G4_data=np.array(sample_Total_G4_data)


			sample_Total_G3_data=[]
			for i in range(Total_G3_data.shape[0]):
				tmp=Total_G3_data[i][0]
				sample_Total_G3_data.append(tmp)
				
			sample_Total_G3_data=np.array(sample_Total_G3_data)


			sample_Total_G2_data=[]
			for i in range(Total_G2_data.shape[0]):
				tmp=Total_G2_data[i][0]
				sample_Total_G2_data.append(tmp)
				
			sample_Total_G2_data=np.array(sample_Total_G2_data)

			sample_Total_G_data=[]
			for i in range(Total_G_data.shape[0]):
				tmp=Total_G_data[i][0]
				sample_Total_G_data.append(tmp)
				
			sample_Total_G_data=np.array(sample_Total_G_data)


			Train_4_data=[]
			for i in range(len(new_correlation4_result_index)):
				tmp=sample_Total_G4_data[new_correlation4_result_index[i]]
				Train_4_data.append(tmp)
			Train_4_data=np.array(Train_4_data)


			Train_3_data=[]
			for i in range(len(new_correlation3_result_index)):
				tmp=sample_Total_G3_data[new_correlation3_result_index[i]]
				Train_3_data.append(tmp)
			Train_3_data=np.array(Train_3_data)


			Train_2_data=[]
			for i in range(len(new_correlation2_result_index)):
				tmp=sample_Total_G2_data[new_correlation2_result_index[i]]
				Train_2_data.append(tmp)
			Train_2_data=np.array(Train_2_data)

			Train_1_data=[]
			for i in range(len(new_correlation1_result_index)):
				tmp=sample_Total_G_data[new_correlation1_result_index[i]]
				Train_1_data.append(tmp)
			Train_1_data=np.array(Train_1_data)

			stage_1_2_data=np.vstack((Train_4_data,Train_3_data))

			stage_1_2_3_data=np.vstack((stage_1_2_data,Train_2_data))

			stage_1_2_3_4_data=np.vstack((stage_1_2_3_data,Train_1_data))

			###########Enter the number of data corresponding to GAN20 for each stage in the range################################
			new_train_label=[]
			for i in range(83):
				new_train_label.append(4)
			for i in range(87):
				new_train_label.append(3)
			for i in range(79):
				new_train_label.append(2)
			for i in range(2):
				new_train_label.append(1)

			###########Load test data################################
			newnew_abnormal_data1=np.load("/Data3/chkwon/Data3_Storage/1212blca/CASE_"+str(j)+"_1014_TEST_Result_blca.txt_stage1.npy")
			newnew_abnormal_data2=np.load("/Data3/chkwon/Data3_Storage/1212blca//CASE_"+str(j)+"_1014_TEST_Result_blca.txt_stage2.npy")
			newnew_abnormal_data3=np.load("/Data3/chkwon/Data3_Storage/1212blca/CASE_"+str(j)+"_1014_TEST_Result_blca.txt_stage3.npy")
			newnew_abnormal_data4=np.load("/Data3/chkwon/Data3_Storage/1212blca/CASE_"+str(j)+"_1014_TEST_Result_blca.txt_stage4.npy")


			newnew_abnormal_data12=np.vstack((newnew_abnormal_data1,newnew_abnormal_data2))
			newnew_abnormal_data123=np.vstack((newnew_abnormal_data12,newnew_abnormal_data3))
			newnew_abnormal_data1234=np.vstack((newnew_abnormal_data123,newnew_abnormal_data4))

			test_label=[]
			for i in range(newnew_abnormal_data1.shape[0]):
				test_label.append(1)
			for i in range(newnew_abnormal_data2.shape[0]):
				test_label.append(2)
			for i in range(newnew_abnormal_data3.shape[0]):
				test_label.append(3)
			for i in range(newnew_abnormal_data4.shape[0]):
				test_label.append(4)
				
			X_train=stage_1_2_3_4_data
	
			X_train=pd.DataFrame(data=X_train)
			
			
			Y_train=new_train_label
			Y_train=np.array(Y_train)
		
			Y_train=to_categorical(Y_train,num_classes=4)
			
			
			X_test=newnew_abnormal_data1234
				
			X_test=pd.DataFrame(data=X_test)

			print("test_label",test_label)
			
			print("test_label",test_label)
			Y_test=test_label
			Y_test=np.array(Y_test)
			Y_test=to_categorical(Y_test,num_classes=4)
			
			print("X_train.shape",X_train.shape)
			print("Y_train.shape",Y_train.shape)
			
			accuracy=CNN1D_evaluate_model(X_train,X_test,Y_train,Y_test,Y_test)
			print("all_score",accuracy)
			print("all_score[0]",accuracy[0])
			print("all_score[1]",accuracy[1])
			print("all_score[2]",accuracy[2])
			print("all_score_length",len(accuracy))
			print("all_score_type",type(accuracy))
			#data="%f \n"%accuracy
			f.write("accuracy")
			f.write(str(accuracy[0]))
			f.write("\n")
			f.write(str(accuracy[1]))
		f.close()
		
	if iter==3:
		mul=100
		f=open("/Data3/chkwon/Data3_Storage/1212_Data/blca_MUL"+str(mul)+"txt",'w')
		for j in range(30):
		################################## GAN 100 ######################################	
			###########Loading GAN100 data corresponding to stage1################################
			Total_correlation_result=np.load('/Data3/chkwon/Data3_Storage/1212blca/CASE_'+str(j)+'_1025_mutation_correlation_stageblca.txt_1MUL'+str(mul)+'.npy')
			Total_G_data=np.load("/Data3/chkwon/Data3_Storage/1212blca/CASE_"+str(j)+"_1025_mutation_Generator_data_stageblca.txt_1MUL"+str(mul)+".npy")
			new_correlation1_result=[]
			line_tmp=[]
			for i in range(len(Total_correlation_result)):
				tmp=Total_correlation_result[i]
				line_tmp.append(i)
				line_tmp.append(tmp)
				new_correlation1_result.append(line_tmp)
				line_tmp=[]
			new_correlation1_result.sort(key=itemgetter(1),reverse=True)

			###########Loading GAN100 data corresponding to stage2################################
			Total_correlation2_result=np.load('/Data3/chkwon/Data3_Storage/1212blca/CASE_'+str(j)+'_1025_mutation_correlation_stageblca.txt_2MUL'+str(mul)+'.npy')
			Total_G2_data=np.load("/Data3/chkwon/Data3_Storage/1212blca/CASE_"+str(j)+"_1025_mutation_Generator_data_stageblca.txt_2MUL"+str(mul)+".npy")
			new_correlation2_result=[]
			line_tmp=[]
			for i in range(len(Total_correlation2_result)):
				tmp=Total_correlation2_result[i]
				line_tmp.append(i)
				line_tmp.append(tmp)
				new_correlation2_result.append(line_tmp)
				line_tmp=[]
			new_correlation2_result.sort(key=itemgetter(1),reverse=True)

			###########Loading GAN100 data corresponding to stage3################################
			Total_correlation3_result=np.load('/Data3/chkwon/Data3_Storage/1212blca/CASE_'+str(j)+'_1025_mutation_correlation_stageblca.txt_3MUL'+str(mul)+'.npy')
			Total_G3_data=np.load("/Data3/chkwon/Data3_Storage/1212blca/CASE_"+str(j)+"_1025_mutation_Generator_data_stageblca.txt_3MUL"+str(mul)+".npy")
			new_correlation3_result=[]
			line_tmp=[]
			for i in range(len(Total_correlation3_result)):
				tmp=Total_correlation3_result[i]
				line_tmp.append(i)
				line_tmp.append(tmp)
				new_correlation3_result.append(line_tmp)
				line_tmp=[]
			new_correlation3_result.sort(key=itemgetter(1),reverse=True)
			###########Loading GAN100 data corresponding to stage4################################
			Total_correlation4_result=np.load('/Data3/chkwon/Data3_Storage/1212blca/CASE_'+str(j)+'_1025_mutation_correlation_stageblca.txt_4MUL'+str(mul)+'.npy')
			Total_G4_data=np.load("/Data3/chkwon/Data3_Storage/1212blca//CASE_"+str(j)+"_1025_mutation_Generator_data_stageblca.txt_4MUL"+str(mul)+".npy")
			new_correlation4_result=[]
			line_tmp=[]
			for i in range(len(Total_correlation4_result)):
				tmp=Total_correlation4_result[i]
				line_tmp.append(i)
				line_tmp.append(tmp)
				new_correlation4_result.append(line_tmp)
				line_tmp=[]
			new_correlation4_result.sort(key=itemgetter(1),reverse=True)
			###########Enter the number of data corresponding to GAN100 for each stage in the range################################
			new_correlation4_result_index=[]
			for i in range(83):
				tmp=new_correlation4_result[i][0]
				new_correlation4_result_index.append(tmp)
				
			new_correlation3_result_index=[]
			for i in range(87):
				tmp=new_correlation3_result[i][0]
				new_correlation3_result_index.append(tmp)
				
			new_correlation2_result_index=[]
			for i in range(79):
				tmp=new_correlation2_result[i][0]
				new_correlation2_result_index.append(tmp)

			new_correlation1_result_index=[]
			for i in range(2):
				tmp=new_correlation1_result[i][0]
				new_correlation1_result_index.append(tmp)

			sample_Total_G4_data=[]
			for i in range(Total_G4_data.shape[0]):
				tmp=Total_G4_data[i][0]
				sample_Total_G4_data.append(tmp)
			sample_Total_G4_data=np.array(sample_Total_G4_data)


			sample_Total_G3_data=[]
			for i in range(Total_G3_data.shape[0]):
				tmp=Total_G3_data[i][0]
				sample_Total_G3_data.append(tmp)
				
			sample_Total_G3_data=np.array(sample_Total_G3_data)


			sample_Total_G2_data=[]
			for i in range(Total_G2_data.shape[0]):
				tmp=Total_G2_data[i][0]
				sample_Total_G2_data.append(tmp)
				
			sample_Total_G2_data=np.array(sample_Total_G2_data)

			sample_Total_G_data=[]
			for i in range(Total_G_data.shape[0]):
				tmp=Total_G_data[i][0]
				sample_Total_G_data.append(tmp)
				
			sample_Total_G_data=np.array(sample_Total_G_data)


			Train_4_data=[]
			for i in range(len(new_correlation4_result_index)):
				tmp=sample_Total_G4_data[new_correlation4_result_index[i]]
				Train_4_data.append(tmp)
			Train_4_data=np.array(Train_4_data)


			Train_3_data=[]
			for i in range(len(new_correlation3_result_index)):
				tmp=sample_Total_G3_data[new_correlation3_result_index[i]]
				Train_3_data.append(tmp)
			Train_3_data=np.array(Train_3_data)


			Train_2_data=[]
			for i in range(len(new_correlation2_result_index)):
				tmp=sample_Total_G2_data[new_correlation2_result_index[i]]
				Train_2_data.append(tmp)
			Train_2_data=np.array(Train_2_data)

			Train_1_data=[]
			for i in range(len(new_correlation1_result_index)):
				tmp=sample_Total_G_data[new_correlation1_result_index[i]]
				Train_1_data.append(tmp)
			Train_1_data=np.array(Train_1_data)

			stage_1_2_data=np.vstack((Train_4_data,Train_3_data))

			stage_1_2_3_data=np.vstack((stage_1_2_data,Train_2_data))

			stage_1_2_3_4_data=np.vstack((stage_1_2_3_data,Train_1_data))

			###########Enter the number of data corresponding to GAN100 for each stage in the range################################
			new_train_label=[]
			for i in range(83):
				new_train_label.append(4)
			for i in range(87):
				new_train_label.append(3)
			for i in range(79):
				new_train_label.append(2)
			for i in range(2):
				new_train_label.append(1)

			###########Load test data################################
			newnew_abnormal_data1=np.load("/Data3/chkwon/Data3_Storage/1212blca/CASE_"+str(j)+"_1014_TEST_Result_blca.txt_stage1.npy")
			newnew_abnormal_data2=np.load("/Data3/chkwon/Data3_Storage/1212blca//CASE_"+str(j)+"_1014_TEST_Result_blca.txt_stage2.npy")
			newnew_abnormal_data3=np.load("/Data3/chkwon/Data3_Storage/1212blca/CASE_"+str(j)+"_1014_TEST_Result_blca.txt_stage3.npy")
			newnew_abnormal_data4=np.load("/Data3/chkwon/Data3_Storage/1212blca/CASE_"+str(j)+"_1014_TEST_Result_blca.txt_stage4.npy")


			newnew_abnormal_data12=np.vstack((newnew_abnormal_data1,newnew_abnormal_data2))
			newnew_abnormal_data123=np.vstack((newnew_abnormal_data12,newnew_abnormal_data3))
			newnew_abnormal_data1234=np.vstack((newnew_abnormal_data123,newnew_abnormal_data4))

			test_label=[]
			for i in range(newnew_abnormal_data1.shape[0]):
				test_label.append(1)
			for i in range(newnew_abnormal_data2.shape[0]):
				test_label.append(2)
			for i in range(newnew_abnormal_data3.shape[0]):
				test_label.append(3)
			for i in range(newnew_abnormal_data4.shape[0]):
				test_label.append(4)
				
			X_train=stage_1_2_3_4_data
	
			X_train=pd.DataFrame(data=X_train)
			
			
			Y_train=new_train_label
			Y_train=np.array(Y_train)
		
			Y_train=to_categorical(Y_train,num_classes=4)
			
			
			X_test=newnew_abnormal_data1234
				
			X_test=pd.DataFrame(data=X_test)

			print("test_label",test_label)
			
			print("test_label",test_label)
			Y_test=test_label
			Y_test=np.array(Y_test)
			Y_test=to_categorical(Y_test,num_classes=4)
			
			print("X_train.shape",X_train.shape)
			print("Y_train.shape",Y_train.shape)
			
			accuracy=CNN1D_evaluate_model(X_train,X_test,Y_train,Y_test,Y_test)
			print("all_score",accuracy)
			print("all_score[0]",accuracy[0])
			print("all_score[1]",accuracy[1])
			print("all_score[2]",accuracy[2])
			print("all_score_length",len(accuracy))
			print("all_score_type",type(accuracy))
			#data="%f \n"%accuracy
			f.write("accuracy")
			f.write(str(accuracy[0]))
			f.write("\n")
			f.write(str(accuracy[1]))
		f.close()