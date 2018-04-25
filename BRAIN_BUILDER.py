import random
from time import clock
from math import exp
from joblib import Parallel, delayed

#mu=0.10		#mutation chance
#muR=0.10	#mutation rate



class Node(object):
	def __init__(self):
		self.Weights=[]
		self.Bias=0.0
		self.Value=0.0
		self.Err=0.0
	
####################################################


def calc_Node(inputs,weights,Bias): #algorithm to calculate the output of a node
	nodeSum=0.
	
	for i in range(len(inputs)):
		nodeSum+=weights[i]*inputs[i]
	nodeSum+=Bias
	#print "SUM ----> \t" + str(nodeSum)
	return(1/(1+exp(-1.*nodeSum)))


def RunBrain(Inputs, Brain, nCores): #algorithm to run a set of inputs through the brain.
	
	for i in range(len(Brain)):
		if i==0:
			inputs_=Inputs
		else:
			inputs_=[]
			for j in range(len(Brain[i-1])):
				inputs_.append(Brain[i-1][j].Value)
		if nCores==1:
			j=1
			
			for aNode in Brain[i] : 
				aNode.Value=calc_Node(inputs_,aNode.Weights,aNode.Bias)
				j+=1
		else:
			Parallel(n_jobs=nCores)(delayed(calc_Node)(inputs_,aNode.Weights,aNode.Bias) for aNode in Brain[i])
	
	#return(Brain[len(Brain)-1])
	

def BackProp(Brain, Key, rate): #backpropagtion algorithm to train the brain.
	i=len(Brain)-1
	while i>0:
		for j in range(len(Brain[i])):
			if i==len(Brain)-1:
				Brain[i][j].Err=Brain[i][j].Value*(1-Brain[i][j].Value)*(Key[j]-Brain[i][j].Value)
			else:
				ErrSum=0.
				for Node in Brain[i+1]:
					ErrSum = ErrSum + Node.Err*Node.Weights[j]
				Brain[i][j].Err = Brain[i][j].Value*(1-Brain[i][j].Value)*ErrSum
		i=i-1;
	
 	for i in range(1,len(Brain)):
 		for j in range(len(Brain[i])):
 			for k in range(len(Brain[i][j].Weights)):
				Brain[i][j].Weights[k]=Brain[i][j].Weights[k]+rate*Brain[i][j].Err*Brain[i-1][k].Value
			Brain[i][j].Bias = Brain[i][j].Bias + rate*Brain[i][j].Err
 				
		

def get_Random():#generates a random number from -1 to 1
	if random.random()>0.5:
		x=random.random()
	else:
		x=-1.*random.random()
	return(x)

def RandomBrain(NumLayers,numNodes,numInputs,numOutputs):#Generates a randomly initialized brain

	HiddenLayers=[]
	#NumLayers=2
	#numNodes=4
	
	#make Right number of layers with right number of Nodes each
	for i in range(NumLayers):
		HiddenLayers.append([])
		for j in range(numNodes):
			HiddenLayers[i].append(Node())
	
	InLayerFlag=True
	for Layer in HiddenLayers :
		for i in range(len(Layer)):
			if InLayerFlag :
				for j in range(numInputs):
					Layer[i].Weights.append(get_Random()*10/numInputs) #get_Random()/(10*numNodes))
				if get_Random()>0.5:
					Layer[i].Bias=-1*get_Random()
				else:
					Layer[i].Bias=get_Random()
			else:
				for j in range(0,numNodes):
					Layer[i].Weights.append(get_Random()*10/numNodes) #get_Random()/(10*numNodes))
				Layer[i].Bias= get_Random()
		InLayerFlag=False
				
	outLayer=[]
	for i in range(numOutputs):
		outLayer.append(Node())
		for j in range(numNodes):
			outLayer[i].Weights.append(get_Random())
		if get_Random()>0.5:
			outLayer[i].Bias=-1*get_Random()
		else:
			outLayer[i].Bias=get_Random()
	Brain=[]
	#Brain.append(InputNodes)
	for Layer in HiddenLayers:
		Brain.append(Layer)
	Brain.append(outLayer)
	return Brain
	


##################################################################################		
def BreedBrains2(Mom,Dad,mu,muR): #Generates a random offspring from Mom and Dad brains. Every feature of the offspring has chance mu (0-1) of mutating by muR (any real)
#Pick the num of hidden layers of Mom or Dad 
	if get_Random() >0.5:
		hLayerNum = len(Dad)-1
	else:
		hLayerNum = len(Mom)-1 #
	
	#chance to mutate
	if get_Random()<mu :
		if get_Random()>0.5:
			hLayerNum+=1
		else:
			if hLayerNum>2:
				hLayerNum-=1
				

#Number of neurons per hidden layer
	if get_Random()>0.5:
		neurNum = len(Dad[0])
	else:
		neurNum=len(Mom[0])
	
	#chance to mutate
	if get_Random()<mu :
		if get_Random()>0.5:
			neurNum+=1
		else:
			if neurNum > 2:
				neurNum-=1
				
	Babby=[] #How is Babby formed?
	
#Initialize the Babby

	#assume mom dad and babby have same number of inputs
	inputNum = len(Mom[0][0].Weights)

	for i in range(hLayerNum):
		Babby.append([])
		for j in range(neurNum):
			Babby[i].append(Node())
			if i==0:
				Babby[i][j].Weights=[None]*inputNum
			else:
				Babby[i][j].Weights=[None]*neurNum
	#output layer (assuming mom and dad have same number of output nodes
	Babby.append([])
	outNeurNum=len(Mom[len(Mom)-1])
	for j in range(outNeurNum):
		Babby[hLayerNum].append(Node())
		Babby[hLayerNum][j].Weights=[None]*neurNum
	
#Assign Values to Weights and Biases of Babby based on parents
	
	#fill in the Hidden Layers
	for i in range(hLayerNum):
		if i==0:
			weightNum=inputNum
		else:
			weightNum=neurNum
			
		for j in range(neurNum):
		
			if get_Random()>0.5:
				try:
					Babby[i][j].Bias=Mom[i][j].Bias
				except IndexError:
					try:
						Babby[i][j].Bias=Dad[i][j].Bias
					except IndexError:
						if get_Random()>0.5:
							Babby[i][j].Bias=-1*get_Random()
						else:
							Babby[i][j].Bias=get_Random()
			else:
				try:
					Babby[i][j].Bias=Dad[i][j].Bias
				except IndexError:
					try:
						Babby[i][j].Bias=Mom[i][j].Bias
					except IndexError:
						if get_Random()>0.5:
							Babby[i][j].Bias=-1*get_Random()
						else:
							Babby[i][j].Bias=get_Random()
			
			for k in range(weightNum):
				if get_Random()>0.5:
					try:
						Babby[i][j].Weights[k]=Mom[i][j].Weights[k]
					except IndexError:
						try:
							Babby[i][j].Weights[k]=Dad[i][j].Weights[k]
						except IndexError:
							Babby[i][j].Weights[k]=get_Random()
				else:
					try:
						Babby[i][j].Weights[k]=Dad[i][j].Weights[k]
					except IndexError:
						try:
							Babby[i][j].Weights[k]=Mom[i][j].Weights[k]
						except IndexError:
							Babby[i][j].Weights[k]=get_Random()
	
#output layer
	momOutI=len(Mom)-1
	dadOutI=len(Dad)-1
	babbyOutI=len(Babby)-1
	weightNum=neurNum		
	for j in range(outNeurNum):
		if get_Random()>0.5:
			try:
				Babby[babbyOutI][j].Bias=Mom[momOutI][j].Bias
			except IndexError:
				try:
					Babby[babbyOutI][j].Bias=Dad[dadOutI][j].Bias
				except IndexError:
					if get_Random()>0.5:
						Babby[babbyOutI][j].Bias=-1*get_Random()
					else:
						Babby[babbyOutI][j].Bias=get_Random()
		else:
			try:
				Babby[babbyOutI][j].Bias=Dad[dadOutI][j].Bias
			except IndexError:
				try:
					Babby[babbyOutI][j].Bias=Mom[momOutI][j].Bias
				except IndexError:
					if get_Random()>0.5:
						Babby[babbyOutI][j].Bias=-1*get_Random()
					else:
						Babby[babbyOutI][j].Bias=get_Random()
	
		for k in range(weightNum):
			if get_Random()>0.5:
				try:
					Babby[babbyOutI][j].Weights[k]=Mom[momOutI][j].Weights[k]
				except IndexError:
					try:
						Babby[babbyOutI][j].Weights[k]=Dad[dadOutI][j].Weights[k]
					except IndexError:
						Babby[babbyOutI][j].Weights[k]=get_Random()
			else:
				try:
					Babby[babbyOutI][j].Weights[k]=Dad[dadOutI][j].Weights[k]
				except IndexError:
					try:
						Babby[babbyOutI][j].Weights[k]=Mom[momOutI][j].Weights[k]
					except IndexError:
						Babby[babbyOutI][j].Weights[k]=get_Random()
				
			#so it has either not worked because we are in a layer that mom or dad does not have or because we are in a node depth that mom or dad does not have. right?
			#Append a node filled with random
			
			#Some Weight matrices may be too long or too short
# mutate	for i
	for layer in Babby:
		for nur in layer:
			if get_Random()<mu:
				if get_Random()<0.5:
					nur.Bias=nur.Bias+muR*nur.Bias
				else:
					nur.Bias=nur.Bias-muR*nur.Bias
				
				for weight in nur.Weights:
					if get_Random()<0.5:
						weight=weight+muR*weight
					else:
						weight=weight-muR*weight
	return(Babby)
	




def CheckBrain(inLen,outLen,Brain): #checks a generated brain for any defects
	
	for i in range(len(Brain)):
		if i==0:
			weightsLen=inLen
			layerLen=len(Brain[0])
		elif i==len(Brain)-1:
			weightsLen=len(Brain[0])
			layerLen=outLen
		else:
			weightsLen=len(Brain[0])
			layerLen=len(Brain[0])

		if len(Brain[i])!= layerLen:
			return(False)
	
		for nur in Brain[i]:
			if len(nur.Weights)!=weightsLen:
				return(False)
	return(True)




if __name__ == '__main__' :
	return 0
	
