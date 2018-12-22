import tensorflow as tf
import keras
import numpy as np
from stanfordcorenlp import StanfordCoreNLP
import csv
import json
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemma_maker = WordNetLemmatizer()
import nltk
import collections
import random
import math



def normalize_text(text):
    text=text.lower()
    text = text.replace(".", '')
    text = text.replace("'", '')
    text = text.replace("`", '')
    text = text.replace("'s", '')
    text = text.replace("/", ' ')
    text = text.replace("\"", ' ')
    text = text.replace("\\", '')
    text =text.replace("@", ' ')
    text = text.replace(",", " ")
    #text =  re.sub(r"\b[a-z]\b", "", text)
    text=text.strip()
    return text

#all_utterances=[]

def refine_word(word):
    rep={}
    final_word=''
    for i in word:
        try:
            rep[i]+=1
        except:
            rep[i]=1
        if rep[i]<2:
            final_word+=i
    word=final_word

word2int={}
int2word={}
words=set([])
sentences=[]


with open('train_data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    big_string=''
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print("gjghtjk")
            line_count += 1
        else:
        	row[1]=normalize_text(row[1])
        	sentence=[]
        	for word in row[1].split():
        		refine_word(word)
        		sentence.append(word)
	        	#if word not in stopwords:
	        	words.add(word)
	        	big_string+=(lemma_maker.lemmatize(word))+ ' '
	        	
        	sentences.append(sentence)
        	print(sentences[-1])
        	line_count += 1
        	
        if line_count>100:
        	break

for i,word in enumerate(words):
    word2int[word] = i
    int2word[i] = word


vocabulary_size = 500000  # Parameter indicating the number of words we'll put in the dictionary
validation_size = 1000  # Size of the validation set
epochs = 20  # Number of epochs we usually start to train with
batch_size = 512  # Size of the batches used in the mini-batch gradient descent
time_steps=8
lstm_size=512
max_length=12
graph=tf.graph()


def convolution(x):
    conv=tf.nn.conv1d

with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, time_steps, max_length))
    tf_train_labels = tf.placeholder(tf.float32,
                                     shape=(batch_size, label_number))
    tf_test_dataset = tf.constant(test_dataset)

    initer = tf.truncated_normal_initializer(stddev=0.01)
    conv_w1 = tf.get_variable(name="conv_w1", dtype=tf.float32, shape=[batch_size, time_steps, max_length], initializer=initer)
    conv_b1 = tf.get_variable(name="conv_b1", dtype=tf.float32, initializer=tf.constant(0.01, shape=[32, ], dtype=tf.float32))
    conv_w2 = tf.get_variable(name="conv_w2", dtype=tf.float32, shape=[7, 7, 32, 64], initializer=initer)
    conv_b2 = tf.get_variable(name="conv_b2", dtype=tf.float32,
                              initializer=tf.constant(0.01, shape=[64, ], dtype=tf.float32))
    conv_w3 = tf.get_variable(name="conv_w3", dtype=tf.float32, shape=[3, 3, 64, 256], initializer=initer)
    conv_b3 = tf.get_variable(name="conv_b3", dtype=tf.float32,
                              initializer=tf.constant(0.01, shape=[256, ], dtype=tf.float32))
    fc_w = tf.get_variable(name='fc_w', dtype=tf.float32, shape=[2304, label_number], initializer=initer)
    fc_b = tf.get_variable(name="fc_b", dtype=tf.float32,
                           initializer=tf.constant(0.0001, shape=[label_number, ], dtype=tf.float32))


    def model(x):
        conv1 = tf.nn.conv2d(x, conv_w1, strides=[1, 1, 1, 1], padding='VALID') + conv_b1
        relu1 = tf.nn.relu(conv1)
        maxp1 = tf.nn.max_pool(relu1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')

        conv2 = tf.nn.conv2d(maxp1, conv_w2, strides=[1, 1, 1, 1], padding="VALID") + conv_b2
        relu2 = tf.nn.relu(conv2)
        maxp2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv3 = tf.nn.conv2d(maxp2, conv_w3, strides=[1, 1, 1, 1], padding='VALID')
        relu3 = tf.nn.relu(conv3)
        maxp3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        shape = maxp3.get_shape().as_list()

        reshape = tf.reshape(maxp3, [shape[0], shape[1] * shape[2] * shape[3]])
        fc = tf.nn.bias_add(tf.matmul(reshape, fc_w), fc_b)
        return fc

    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    train_prediction = tf.nn.softmax(logits=logits)

    test_prediction = tf.nn.softmax(model(tf_test_dataset))


    #'''
    # input data  words_in_dataset = tf.placeholder(tf.float32, [time_steps, batch_size, num_features])
    #lstm=tf.contrib.rnn.BasicLSTMCell(lstm_size)
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size,time_steps,num_features])
    #state = lstm.zero_state(batch_size, dtype=tf.float32)
    conw1=tf.get_variable(name="convw1")

    def model(x):
        conv1 = tf.nn.conv2d(x, conv_w1, strides=[1, 1, 1, 1], padding='VALID') + conv_b1
        relu1 = tf.nn.relu(conv1)
        #maxp1 = tf.nn.max_pool(relu1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')

        conv2 = tf.nn.conv2d(maxp1, conv_w2, strides=[1, 1, 1, 1], padding="VALID") + conv_b2
        relu2 = tf.nn.relu(conv2)
        #maxp2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv3 = tf.nn.conv2d(maxp2, conv_w3, strides=[1, 1, 1, 1], padding='VALID')
        relu3 = tf.nn.relu(conv3)
        #maxp3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        shape = maxp3.get_shape().as_list()

        reshape = tf.reshape(maxp3, [shape[0], shape[1] * shape[2] * shape[3]])
        fc = tf.nn.bias_add(tf.matmul(reshape, fc_w), fc_b)
        return fc
    #'''



tf.reset_default_graph()
# Create some variables.
v1 = tf.get_variable("v1", [3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", [5], initializer = tf.zeros_initializer)



# Add ops to save and restore only `v2` using the name "v2"
saver = tf.train.Saver({"v2": v2})

# Use the saver object normally after that.
with tf.Session() as sess:
  # Initialize v1 since the saver will not.
  v1.initializer.run()
  #tf.saved_model.loader.load(sess, [tag_constants.TRAINING],"my_test_model99999-1000.meta" )
  saver.export_meta_graph('my_test_model99999-1000.meta')
  print(v1.eval(session=sess))

