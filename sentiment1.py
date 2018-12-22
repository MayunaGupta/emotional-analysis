from Naked.toolshed.shell import execute_js, muterun_js

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
import sys

file=open("output.txt","w+")
file_curve=open("output3.txt","w+")
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

			sentiment=normalize_text(row[0])
			file_curve.write(sentiment+'\n')
			row[1]=normalize_text(row[1])

			sentence=''
			for word in row[1].split():
				refine_word(word)
				sentence+= word
				sentence+=" "
				#if word not in stopwords:
				words.add(word)
				big_string+=(lemma_maker.lemmatize(word))+ ' '
			#sys.stdout.write( sentence+ '\n')
			file_curve.write(sentiment+",")
			file.write(sentence+"\n")
			sentences.append(sentence)
			#success = execute_js('server.js')
			#print("js challing")
			print(sentences[-1])
			line_count += 1
			
		if line_count>100:
			break

for i,word in enumerate(words):
	word2int[word] = i
	int2word[i] = word

file.close()


file_curve.close()
success = execute_js('server.js')




#print(sentences)

