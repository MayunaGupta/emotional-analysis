from stanfordcorenlp import StanfordCoreNLP
import csv
import json
from watson_developer_cloud import ToneAnalyzerV3



nlp = StanfordCoreNLP(r'/home/mg/Documents/courses/emotional-analysis/stanford-corenlp-full-2018-02-27')
sentence = 'Guangdong University of Foreign Studies is located in Guangzhou.'
print 'Tokenize:', nlp.word_tokenize(sentence)
print 'Part of Speech:', nlp.pos_tag(sentence)
print 'Named Entities:', nlp.ner(sentence)
print 'Constituency Parsing:', nlp.parse(sentence)
print 'Dependency Parsing:', nlp.dependency_parse(sentence)

nlp.close() # Do not forget to close! The backend server will consume a lot memery.
dict={}

'''
all_utterances=[]
with open('train_data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
    	dict={}
        if line_count == 0:
            print("gjghtjk")
            line_count += 1
        else:
        	dict['text']=row[1]
        	dict['emotion']=row[0]
        	all_utterances.append(dict)
        	print(row)
        	line_count += 1
    print('Processed lines.')



'''
'''
tone_analyzer = ToneAnalyzerV3(
    version='2018-10-01',
    username='4555755a-86b2-43d3-bb43-94fca1dbe55b',
    password='W2vohFepStr2',
    url='https://gateway.watsonplatform.net/tone-analyzer/api'


  "url": "https://gateway.watsonplatform.net/tone-analyzer/api",
  "username": "4555755a-86b2-43d3-bb43-94fca1dbe55b",
  "password": "W2vohFepStr2"
cf cups Personality-Insights-Std -p "\"4555755a-86b2-43d3-bb43-94fca1dbe55b\":\"W2vohFepStr2\""
                                                                    
)
utterances = [
    {
        "text": "Hello, I'm having a problem with your product.",
        "user": "customer"
    },
    {
        "text": "OK, let me know what's going on, please.",
        "user": "agent"
    },
    {
        "text": "Well, nothing is working :(",
        "user": "customer"
    },
    {
        "text": "Sorry to hear that.",
        "user": "agent"
    }
]

utterance_analyses = tone_analyzer.tone_chat(utterances).get_result()
print(json.dumps(utterance_analyses, indent=2))
#print(utterance_analyses)
''' 