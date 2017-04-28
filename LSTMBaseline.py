import json, os, nltk
import numpy as np
import dynet as dn
from gensim.models.keyedvectors import KeyedVectors

word2VecFile = "/GW/D5data-1/kpopat/SnopesDeepLearning/resources/gloveWiki6B/word2vec.6B.300d.txt"
JSONFiles = "/GW/D5data-1/kpopat/SnopesDeepLearning/resources/test"

missingWordFilePath = "/GW/D5data-1/kpopat/SnopesDeepLearning/Workspace/missing_word.txt"

missingWordFile = open(missingWordFilePath,"w") 
missingWords = set()

vocab = {}
vectors = []

def prepareEmbeddings(vectorFile):
	with open(word2VecFile) as f:
		f.readline()
		for i, line in enumerate(f):
			fields = line.strip().split(" ")
			vocab[fields[0]] = i
			vectors.append(list(map(float, fields[1:])))


def prepare_dataset(path_to_folder):
	trainData = []
	trainLabels = []
	#iterate over all json file -- one json per claim
	for jsonFileName in os.listdir(path_to_folder):
		if not jsonFileName.endswith(".json"): 
			continue
		else:
			#load the json file
			with open(os.path.join(path_to_folder, jsonFileName)) as jsonFile:
				jsonData = json.load(jsonFile)
			
			print("Processing: "+jsonData["Claim_ID"])
			trainLabels.append(jsonData["Credibility"])
	
			docVectorCollection = []
			
			#for each search result page in json
			for searchPage in jsonData["Google Results"]:
				
				#for each search result document in the search page
				for searchResult in searchPage["results"]:
					
					#read the document text and convert it to small case
					docText = searchResult["doc_text"]
					
					#skip the empty documents and snopes web pages
					if docText == "" or searchResult["domain"] == "www.snopes.com":
						continue
					
					docText = docText.lower()
					docVector = []
	  
					#for each word in the document get the pretrained embedding vectors
					for word in nltk.word_tokenize(docText):
						if word in vocab:
							docVector.append(vocab[word])
						else:
							missingWords.add(word)
	    
					docVectorCollection.append(docVector)
	    
			print("\tAdding {} documents".format(len(docVectorCollection)))			
			trainData.append(docVectorCollection)
	
	#dump missing words
	for missedWord in missingWords:
		missingWordFile.write(missedWord+"\n")
	missingWordFile.close()
	
	return trainData, trainLabels


print("Loading Embeddings")
prepareEmbeddings(word2VecFile)
print("Embeddings Loaded!")

print("Preparing data")	
trainData, trainLabels = prepare_dataset(JSONFiles)
print("Data prepared!")


## DEFINE MODEL PARAMETERS
LSTM_NUM_OF_LAYERS = 1
STATE_SIZE = 10
EMBEDDINGS_SIZE = len(vectors[0])
NUM_OF_CLASSES = 2
VOCAB_SIZE = len(vectors)

model = dn.Model()

input_lookup = model.add_lookup_parameters((VOCAB_SIZE, EMBEDDINGS_SIZE))
input_lookup.init_from_array(np.array(vectors))
#print(input_lookup[vocab["hello"]].value())

lstm = dn.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)

output_w = model.add_parameters((NUM_OF_CLASSES, STATE_SIZE))
output_b = model.add_parameters((NUM_OF_CLASSES))


## RUN LSTM FOR EACH DOCUMENT
for related_articles,cred_label in zip(trainData, trainLabels):
	for article in related_articles:
		## RENEW COMPUTATION GRAPH ---- NEED TO CHECK THE IMPACT OF HAVING IT HEAR
		dn.renew_cg()
		expressionSequence = []

		## GET THE EXPRESSION SEQUENCE FOR THE ARTICLE
		for word in article:
			expressionSequence.append(input_lookup[word])

		##INITIALIZE STATE FOR LSTM
		state = lstm.initial_state()
		
		## GET THE FINAL OUTPUT VECTOR OF LENGTH STATE_SIZE 
		output_vec = state.transduce(expressionSequence)[-1]
		print(output_vec.value())
			