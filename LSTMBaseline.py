import json, os, pickle
import numpy as np
from numpy import mean, argmax
import _dynet as dn
#from gensim.models.keyedvectors import KeyedVectors

dyparams = dn.DynetParams()
dyparams.set_mem(12000)
dyparams.set_random_seed(666)
dyparams.init()

np.random.seed(7)

true_class_weight = 2.8
false_class_weight = 1

word2VecFile = "/GW/D5data-1/kpopat/SnopesDeepLearning/resources/gloveWiki6B/word2vec.6B.300d.txt"
#JSONFiles = "/GW/D5data-1/kpopat/SnopesDeepLearning/resources/Snopes+Web_json"
dataPklFiles = "/GW/D5data-1/kpopat/SnopesDeepLearning/resources/Snopes+Web_pickle"
#dataPklFiles = "/GW/D5data-1/kpopat/SnopesDeepLearning/resources/test"

#missingWordFilePath = "/GW/D5data-1/kpopat/SnopesDeepLearning/Workspace/missing_word.txt"

vocab = {}
vectors = []

def prepareEmbeddings(vectorFile):
	print("Loading Embeddings")
	with open(word2VecFile) as f:
		f.readline()
		for i, line in enumerate(f):
			fields = line.strip().split(" ")
			vocab[fields[0]] = i
			vectors.append(list(map(float, fields[1:])))
	print("Embeddings Loaded!")


def get_probs(related_articles):
	
	input_vector_dense_layer = dn.vecInput(30*STATE_SIZE)
	
	output_vector_list = []
	## RUN LSTM FOR EACH DOCUMENT
	
	dn.renew_cg()
	
	for article in related_articles[:30]:
		## RENEW COMPUTATION GRAPH ---- NEED TO CHECK THE IMPACT OF HAVING IT HERE
		
		expressionSequence = []
		
		## GET THE EXPRESSION SEQUENCE FOR THE ARTICLE
		for word in article:
			expressionSequence.append(input_lookup[word])
		
		if len(expressionSequence)==0:
			continue
		##INITIALIZE STATE FOR LSTM
		state = lstm.initial_state()
		
		## GET THE FINAL OUTPUT VECTOR OF LENGTH STATE_SIZE 
		output_vec = state.transduce(expressionSequence)[-1]
		#print(output_vec.value()
		
		output_vector_list.append(output_vec)
	
	#pad the input to dense layers if the number of articles are <30
	if(len(output_vector_list)<30):
		for i in range(30-len(output_vector_list)):
			output_vector_list.append(dn.vecInput(STATE_SIZE))
		
	#concatenate LSTM output of all the documents
	input_vector_dense_layer = dn.concatenate(output_vector_list)
	
	w = dn.parameter(w1)
	b = dn.parameter(b1)
	#Run through dense layer
	output_hl_1 = dn.tanh((w*input_vector_dense_layer)+b)
	
	w = dn.parameter(w2)
	b = dn.parameter(b2)
	#Run through dense layer
	output_hl_2 = dn.tanh((w*output_hl_1)+b)
	
	w = dn.parameter(output_w)
	b = dn.parameter(output_b)
	#Run through softmax layer
	output_scores = dn.softmax(w*output_hl_2+b)
	
	return output_scores
	

def train():
	trainer = dn.AdamTrainer(model)
	print("Training started")
	
	
	#iterate over all json file -- one json per claim
	for pklFileName in os.listdir(dataPklFiles):
		if not pklFileName.endswith(".p"): 
			continue
		else:
			#load the json file
			articles,cred_label = pickle.load(open(os.path.join(dataPklFiles, pklFileName),'rb'))
			
			print("Processing: "+pklFileName)
				    
			print("\tNumber of documents",len(articles))			
			
			print("\tTraining using this instance")		
			
			probs = get_probs(articles)
			#print(probs.value())
			
			#boost the loss for true claims to handle the data imbalance
			if cred_label == 1:
				loss = dn.scalarInput(false_class_weight) * dn.pickneglogsoftmax(probs, cred_label)
			else:
				loss = dn.scalarInput(true_class_weight) * dn.pickneglogsoftmax(probs, cred_label)
				
			loss_val = loss.value()
			
			print("\tLoss: ",loss_val)
			loss.backward()
			trainer.update()
			
	print("Training done")

	
def validate():
	print('starting validation')
	acc = []
	false_acc = []
	true_acc = []
	num_true = 0
	num_false = 0
	num_claims = 0
	
	
	for pklFileName in os.listdir(dataPklFiles):
		if not pklFileName.endswith(".p"): 
			continue
		else:
			#load the json file
			articles,cred_label = pickle.load(open(os.path.join(dataPklFiles, pklFileName),'rb'))
			
			#print("Evaluating: "+pklFileName)
				    
			#print("\tNumber of documents",len(articles))			
			
			#print("\tTraining using this instance")		
			
			probs = get_probs(articles).npvalue()
			num_claims = num_claims + 1
			
			if cred_label == np.argmax(probs):
				acc.append(1)
				if cred_label == 1:
					false_acc.append(1)
					num_false = num_false + 1
				else:
					true_acc.append(1)
					num_true = num_true + 1
			else:
				acc.append(0)
				if cred_label == 1:
					false_acc.append(0)
					num_false = num_false + 1
				else:
					true_acc.append(0)
					num_true = num_true + 1
			
	#print(acc)
	print("++++++++++Results+++++++++++")
	print("Number of Claims: ", num_claims)
	print("Number of True Claims: ", num_true)
	print("Number of False Claims: ", num_false)
	
	print('accuracy: ', mean(acc))
	print('false class accuracy: ', mean(false_acc))
	print('true class accuracy: ', mean(true_acc))


prepareEmbeddings(word2VecFile)

#trainData, trainLabels = prepare_dataset(JSONFiles)

print("Initializing Model")
## DEFINE MODEL PARAMETERS
LSTM_NUM_OF_LAYERS = 1
STATE_SIZE = 10
EMBEDDINGS_SIZE = len(vectors[0])
NUM_OF_CLASSES = 2
VOCAB_SIZE = len(vectors)

## DENSE LAYER PARAMS
HIDDEN_LAYER_SIZE_1 = 20
HIDDEN_LAYER_SIZE_2 = 20

model = dn.Model()

input_lookup = model.add_lookup_parameters((VOCAB_SIZE, EMBEDDINGS_SIZE))
input_lookup.init_from_array(np.array(vectors))
#print(input_lookup[vocab["hello"]].value())

w1 = model.add_parameters((HIDDEN_LAYER_SIZE_1, 30*STATE_SIZE))
b1 = model.add_parameters((HIDDEN_LAYER_SIZE_1))

w2 = model.add_parameters((HIDDEN_LAYER_SIZE_2, HIDDEN_LAYER_SIZE_1))
b2 = model.add_parameters((HIDDEN_LAYER_SIZE_2))

output_w = model.add_parameters((NUM_OF_CLASSES, HIDDEN_LAYER_SIZE_2))
output_b = model.add_parameters((NUM_OF_CLASSES))

lstm = dn.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)
print("Model Initialized!")

train()

validate()
