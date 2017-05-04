import json, os, nltk, pickle

word2VecFile = "/GW/D5data-1/kpopat/SnopesDeepLearning/resources/gloveWiki6B/word2vec.6B.300d.txt"
JSONFiles = "/GW/D5data-1/kpopat/SnopesDeepLearning/resources/Snopes+Web_json"
#JSONFiles = "/GW/D5data-1/kpopat/SnopesDeepLearning/resources/test"

dataPklPath = "/GW/D5data-1/kpopat/SnopesDeepLearning/resources/Snopes+Web_pickle"

missingWordFilePath = "/GW/D5data-1/kpopat/SnopesDeepLearning/Workspace/missing_word.txt"

vocab = {}


def prepareVocab():
	print("Loading Vocab")
	with open(word2VecFile) as f:
		f.readline()
		for i, line in enumerate(f):
			fields = line.strip().split(" ")
			vocab[fields[0]] = i
			#vectors.append(list(map(float, fields[1:])))
	print("Vocab Loaded!")


def prepare_dataset():
	print("Preparing data")	
	
	missingWordFile = open(missingWordFilePath,"w") 
	missingWords = set()
	
	numClaims = 0
	#iterate over all json file -- one json per claim
	for jsonFileName in os.listdir(JSONFiles):
		if not jsonFileName.endswith(".json"): 
			continue
		else:
			#load the json file
			with open(os.path.join(JSONFiles, jsonFileName)) as jsonFile:
				jsonData = json.load(jsonFile)
			
			print("Processing: "+jsonData["Claim_ID"])
			
			if(jsonData["Credibility"] == "true"):
				cred_label = 0
			else:
				cred_label = 1
	
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
					
					##ONLY FIRST 50,000 CHARACTERS -- IS IT OKAY?
					docText = docText[:50000].lower()
					docText = docText.lower()
					docVector = []
	  
					#for each word in the document get the pretrained embedding vectors
					for sent in nltk.sent_tokenize(docText):
						for word in nltk.word_tokenize(sent):
							if word in vocab:
								docVector.append(vocab[word])
							else:
								missingWords.add(word)
	    
					docVectorCollection.append(docVector)
	    
			print("\tAdding {} documents".format(len(docVectorCollection)))
			
			filePath = os.path.join(dataPklPath, os.path.splitext(jsonFileName)[0]+".p");
			if(not os.path.isfile(filePath)):
				pickle.dump((docVectorCollection,cred_label), open(filePath, "wb" ))
				
			numClaims = numClaims + 1
	
	#dump missing words
	for missedWord in missingWords:
		missingWordFile.write(missedWord+"\n")
	missingWordFile.close()
	
	print("Data prepared!")
	print("Loaded claims:",numClaims)
	
prepareVocab()
prepare_dataset()
 
