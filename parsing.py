import tensorflow as tf
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
stopwords = nltk.corpus.stopwords.words('english')
print(stopwords)


vocabulary_size=500000
#wordsList = np.load('wordsList.npy')
#print('Loaded the word list!')
#wordsList = wordsList.tolist() #Originally loaded as numpy array
#wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
#wordVectors = np.load('wordVectors.npy')
#print ('Loaded the word vectors!')

nlp = StanfordCoreNLP(r'/home/mg/Documents/courses/emotional-analysis/stanford-corenlp-full-2018-02-27')
f=open("output_parsing.txt","a")

'''
temp=nlp.word_tokenize(row[1])
nlp.pos_tag(row[1])
nlp.ner(row[1])
nlp.parse(row[1])	
nlp.dependency_parse(row[1])	
			#dict['text']=row[1]
			#dict['emotion']=row[0]

			#all_utterances.append(dict)
			#print(row)
			row[1]=normalize_text(row[1])
			storing_parsing['words :'+repr(line_count)]=nlp.word_tokenize(row[1])
			storing_parsing['postag'+str(line_count)]=nlp.pos_tag(row[1])
			storing_parsing['Parsing:'+repr(line_count)]= nlp.parse(row[1])
			#print(storing_parsing)


 from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
docs=[]
for document in documents:
	text=normalize_text(document)
	nl_text=''
	for word in word_tokenize(text):
		if word not in stopwords:
			nl_text+=(lemmatizer.lemmatize(word))+' '
	docs.append(nl_text)
'''
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
storing_parsing={}

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
			storing_parsing['words :'+repr(line_count)]=nlp.word_tokenize(sentence)
			storing_parsing['postag'+str(line_count)]=nlp.pos_tag(sentence)
			storing_parsing['Parsing:'+repr(line_count)]= nlp.parse(sentence)
			line_count += 1

			#print(big_string)
		if line_count>100:
			break

for i,word in enumerate(words):
	word2int[word] = i
	int2word[i] = word


#print(storing_parsing)
print(word2int)
print(int2word)

#f.write(big_string)
#f.write(str(storing_parsing))
#f.write(str(word2int))
#f.write(str(int2word))
print('Processed lines.')


f.close()	


def build_dataset(words,word2int,int2word):
	# create counts list, set counts for "UNK" token to -1 (undefined)
	count = [['UNK', -1]]
	# add counts of the 49,999 most common tokens in 'words'
	count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
	# create the dictionary data structure
	
	data = []
	# keep track of the number of "UNK" token occurrences
	unk_count = 0
	# for each word in our list of words
	for sentence in sentences:
		for word in sentence:
			# if its in the dictionary, get its index
			if word in words:
				index = word2int[word]
			# otherwise, set the index equal to zero (index of "UNK") and increment the "UNK" count
			else:
				index = 0  # dictionary['UNK']
				unk_count += 1
			# append its index to the 'data' list structure
			data.append(index)
		# set the count of "UNK" in the 'count' data structure
	count[0][1] = unk_count
	#ictionary, and inverted dictionary
	return data, count, word2int, int2word

# build the datset
data, count, dictionary, reverse_dictionary = build_dataset(words,word2int,int2word)

print(data)

del words,word2int,int2word
data_index = 0

# generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
	global data_index
	# make sure our parameters are self-consistent
	assert batch_size % num_skips == 0
	assert num_skips <= 2 * skip_window
	# create empty batch ndarray using 'batch_size'
	batch = np.ndarray(shape=(batch_size), dtype=np.int32)
	# create empty labels ndarray using 'batch_size'
	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
	# [ skip_window target skip_window ]
	span = 2 * skip_window + 1
	# create a buffer object for prepping batch data
	buffer = collections.deque(maxlen=span)
	# for each element in our calculated span, append the datum at 'data_index' and increment 'data_index' moduli the amount of data
	for _ in range(span):
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	# loop for 'batch_size' // 'num_skips'
	for i in range(batch_size // num_skips):
		 # target label at the center of the buffer
		target = skip_window
		targets_to_avoid = [skip_window]
		# loop for 'num_skips'
		for j in range(num_skips):
			# loop through all 'targets_to_avoid'
			while target in targets_to_avoid:
				# pick a random index as target
				target = random.randint(0, span - 1)
			# put it in 'targets_to_avoid'
			targets_to_avoid.append(target)
			# set the skip window in the minibatch data
			batch[i * num_skips + j] = buffer[skip_window]
			# set the target in the minibatch labels
			labels[i * num_skips + j, 0] = buffer[target]
		# add the data at the current 'data_index' to the buffer
		buffer.append(data[data_index])
		# increment 'data_index'
		data_index = (data_index + 1) % len(data)
	# return the minibatch data and corresponding labels
	return batch, labels

# get a minibatch
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)

print(batch)

for i in range(8):
	print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])


##======================================tf part


batch_size = 64
embedding_size = 64 # dimension of the embedding vector
skip_window = 1 # how many words to consider to left and right
num_skips = 2 # how many times to reuse an input to generate a label

# we choose random validation dataset to sample nearest neighbors
# here, we limit the validation samples to the words that have a low
# numeric ID, which are also the most frequently occurring words
valid_size = 16 # size of random set of words to evaluate similarity on
valid_window = 100 # only pick development samples from the first 'valid_window' words
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64 # number of negative examples to sample

# create computation graph
graph = tf.Graph()


with graph.as_default():
	# input data
	train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
	train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
	valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
	
	# operations and variables
	# look up embeddings for inputs
	embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
	embed = tf.nn.embedding_lookup(embeddings, train_inputs)

	# construct the variables for the NCE loss
	nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
	nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

	# compute the average NCE loss for the batch.
	# tf.nce_loss automatically draws a new sample of the negative labels each time we evaluate the loss.
	loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,labels=train_labels, inputs=embed, num_sampled=num_sampled, num_classes=vocabulary_size))
	
	# construct the SGD optimizer using a learning rate of 1.0
	optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

	# compute the cosine similarity between minibatch examples and all embeddings
	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
	normalized_embeddings = embeddings / norm
	valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
	similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

	# add variable initializer
	init = tf.initialize_all_variables()


	# steps to train the model

num_steps = 100000
steplist=[]
losslist=[]
with tf.device("/cpu:0"):
	with tf.Session(graph=graph) as session:
			saver=tf.train.Saver()
			init.run()
			print('initialized.')
			average_loss = 0
			for step in xrange(num_steps):
				# generate a minibatch of training data
				batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
				feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

				# we perform a single update step by evaluating the optimizer operation (including it
				# in the list of returned values of session.run())
				_, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
				average_loss += loss_val

				# print average loss every 2,000 steps
				if step % 2000 == 0:
					steplist.append(step)
					if step > 0:
						average_loss /= 2000
						# the average loss is an estimate of the loss over the last 2000 batches.
						print("Average loss at step ", step, ": ", average_loss)
						losslist.append(average_loss)
						average_loss = 0
					saver.save(session, '/home/mg/Documents/courses/emotional-analysis/my_test_model'+str(step),global_step=step)

			final_embeddings = normalized_embeddings.eval()
			print(final_embeddings)
# computing cosine similarity (expensive!)
#if step % 10000 == 0:
#    sim = similarity.eval()s
#    for i in xrange(valid_size):
#        # get a single validation sample
#        valid_word = reverse_dictionary[valid_examples[i]]
#        # number of nearest neighbors
#        top_k = 8
#       # computing nearest neighbors
#       nearest = (-sim[i, :]).argsort()[1:top_k + 1]
#       log_str = "nearest to %s:" % valid_word
#       for k in xrange(top_k):
#           close_word = reverse_dictionary[nearest[k]]
#           log_str = "%s %s," % (log_str, close_word)
#       print(log_str)

		

		






