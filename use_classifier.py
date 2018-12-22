# probability threshold
ERROR_THRESHOLD = 0.2
# load our calculated synapse values
synapse_file = 'synapses.json' 
with open(synapse_file) as data_file: 
    synapse = json.load(data_file) 
    synapse_0 = np.asarray(synapse['synapse0']) 
    synapse_1 = np.asarray(synapse['synapse1'])

def classify(sentence, show_details=False):

    results = think(sentence, show_details)
    for i,r in enumerate(results):
    	r=r/sum(results)
    results = [[i,r] for i,r in enumerate(results)]# if r> ERROR_THRESHOLD] 
    results.sort(key=lambda x: x[1], reverse=True) 
    return_results =[[classes[r[0]],r[1]] for r in results]
    print ("%s \n classification: %s" % (sentence, return_results))
    return return_results

classify("sudo make me a sandwich")
classify("how are you today?")
classify("talk to you tomorrow")
classify("who are you?")
classify("make me some lunch")
print ()
classify("you are a random person", show_details=True)
print()
classify("i live my life like a princ", show_details=True)