import json, os, nltk
from gensim.models.keyedvectors import KeyedVectors

word2VecFile = "/GW/D5data-1/kpopat/SnopesDeepLearning/resources/gloveWiki6B/word2vec.6B.300d.txt"
JSONFiles = "/GW/D5data-1/kpopat/SnopesDeepLearning/resources/test"

missingWordFilePath = "/GW/D5data-1/kpopat/SnopesDeepLearning/Workspace/missing_word.txt"

print("Loading Embeddings")
embeddings = KeyedVectors.load_word2vec_format(word2VecFile, binary=False)
print("Embeddings Loaded!")

missingWordFile = open(missingWordFilePath,"w") 
missingWords = set()

def prepare_dataset(path_to_folder):
	trainData = []
	trainLables = []
	#iterate over all json file -- one json per claim
	for jsonFileName in os.listdir(path_to_folder):
		if not jsonFileName.endswith(".json"): 
			continue
		else:
			#load the json file
			with open(os.path.join(path_to_folder, jsonFileName)) as jsonFile:
				jsonData = json.load(jsonFile)
			
			print("Processing: "+jsonData["Claim_ID"])
			trainLables.append(jsonData["Credibility"])
	
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
						if word in embeddings:
							docVector.append(embeddings[word])
						else:
							missingWords.add(word)
	    
					docVectorCollection.append(docVector)
	    
			print("\tAdding {} documents".format(len(docVectorCollection)))			
			trainData.append(docVectorCollection)
	
	#dump missing words
	for missedWord in missingWords:
		missingWordFile.write(missedWord+"\n")
	missingWordFile.close()
	
	return trainData, trainLables
	

print("Preparing data")	
trainData, trainLables = prepare_dataset(JSONFiles)
print("Data prepared!")


print("Number of claims: {}".format(len(trainData)))
#print(len(trainData[0]))
#print(len(trainData[1]))
#print(len(trainData[2]))
#print(len(trainData[0][0]))
#print(trainData[0][0][0])
#print(len(trainLables))
	
	
	      
	    
	    
	 





  
  
 

    
