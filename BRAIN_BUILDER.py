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
	
####################################################33
def calc_Node(inputs,weights,Bias):
	nodeSum=0.
	
	for i in range(len(inputs)):
		nodeSum+=weights[i]*inputs[i]
	nodeSum+=Bias
	#print "SUM ----> \t" + str(nodeSum)
	return(1/(1+exp(-1.*nodeSum)))


def RunBrain(Inputs, Brain, nCores):
	
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
	

def BackProp(Brain, Key, rate):
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
 				
		

def get_Random():
	if random.random()>0.5:
		x=random.random()
	else:
		x=-1.*random.random()
	return(x)

def RandomBrain(NumLayers,numNodes,numInputs,numOutputs):#R

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
	
	
def BreedBrains(Mom,Dad):#Creates a random mix of two brains. 

	#Pick the length of Mom or Dad 
	if get_Random() >0.5:
		maxLength = len(Dad)-1
	else:
		maxLength = len(Mom)-1 #only hidden layers
	
	#print len(Dad),len(Mom)
	
	
	#chance to mutate
	if get_Random()<mu :
		if get_Random()>0.5:
			maxLength+=1
		else:
			if maxLength>2:
				maxLength-=1
				
	###print 'Babby Len: '+str(maxLength)+" + output Layer"	

	#print maxLength
	
	if get_Random()>0.5:
		maxNNum = len(Dad[0])
	else:
		maxNNum=len(Mom[0])
	
	#chance to mutate
	if get_Random()<mu :
		if get_Random()>0.5:
			maxNNum+=1
		else:
			if maxNNum > 2:
				maxNNum-=1
	###print 'Babby Depth: '+str(maxNNum)
	
	Babby = [] #How is babby formed?
	#Babby.append(InputNodes)
	
	
	###print "Building Babby..."
	for i in range(0,maxLength):
		Babby.append([])
		###print "----->Layer: "+str(len(Babby))
			
		if i>len(Dad)-1 and i<=len(Mom)-1:#if we've outgrown dad but not mom
			##print 1
			Babby[i]=Mom[i]
			
		
			while len(Babby[i])<maxNNum:
				try:
					Babby[i].append(Babby[i][random.randrange(0,len(Babby[i])-1)])
				except:
					Babby[i].append(Babby[i][0])
		
			if len(Babby[i])>maxNNum:
				del Babby[maxNNum:]
				
		elif i<=len(Dad)-1 and i>len(Mom)-1:#visa versa
			###print 2
			Babby[i]=Dad[i]
			
			
			while len(Babby[i])<maxNNum:
				try:
					Babby[i].append(Babby[i][random.randrange(0,len(Babby[i])-1)])
				except:
					Babby[i].append(Babby[i][0])
					
			if len(Babby[i])>maxNNum:
				del Babby[maxNNum:]
		
		elif i>len(Dad)-1 and i>len(Mom)-1:#if we've mutated beyond both
			###print 3
			Babby[i]=Babby[random.randrange[1,len(Babby)]]
		
		else:
			###print 4
			
			for j in range(0,maxNNum):
				Babby[i].append(Node())
				###print "Node(): "+str(len(Babby[i]))
				
				###print len(Dad[i])-1,len(Mom[i])-1,j

				if j>len(Dad[i])-1 and j <=len(Mom[i])-1: #If we've outgrown dad but not mom use moms weight list 
					###print 'A'
					Babby[i][j]=Mom[i][j]
					
					
					if i==0:
						maxk=len(Mom[0][0].Weights)
					else:
						maxk=len(Babby[0])
					
					while len(Babby[i][j].Weights)< maxk:	
						Babby[i][j].Weights.append(get_Random())
						
					if len(Babby[i][j].Weights) > len(Babby[0]):
						del Babby[i][j].Weights[len(Babby[0]):]
					
					
				elif j<=len(Dad[i])-1 and j >len(Mom[i])-1: # and visa versa
					###print 'B'
					Babby[i][j] = Dad[i][j]
					
										
					if i==0:
						maxk=len(Mom[0][0].Weights)
					else:
						maxk=len(Babby[0])
					
					while len(Babby[i][j].Weights)< maxk:	
						Babby[i][j].Weights.append(get_Random())
					
					if len(Babby[i][j].Weights) > len(Babby[0]):
						del Babby[i][j].Weights[len(Babby[0]):]

						
				elif j>len(Dad[i])-1 and j>len(Mom[i])-1: #if we've mutated beyond both of them
					###print 'C'
					Babby[i][j] = Babby[i][random.randrange(0,len(Babby[i]))]
				else: 
					if i==0:
						krange=len(Mom[0][0].Weights)#Special case for input layer
					else:
						krange=len(Babby[0])
					###print 'D'
					for k in range(krange):
						
						if k> len(Mom[i][j].Weights)-1 and k<= len(Dad[i][j].Weights)-1:
							Babby[i][j].Weights.append(Dad[i][j].Weights[k])
							Babby[i][j].Bias=Dad[i][j].Bias
						elif k <= len(Mom[i][j].Weights)-1 and k > len(Dad[i][j].Weights)-1:
							Babby[i][j].Weights.append(Mom[i][j].Weights[k])
							Babby[i][j].Bias=Mom[i][j].Bias
						elif  k > len(Mom[i][j].Weights)-1 and k > len(Dad[i][j].Weights)-1:
							Babby[i][j].Weights.append(get_Random())
							if get_Random()>0.5:
								Babby[i][j].Bias=-1*get_Random()
							else:
								Babby[i][j].Bias=get_Random()
							
						else:
							if get_Random() >0.5 :
								Babby[i][j].Weights.append(Dad[i][j].Weights[k])
								Babby[i][j].Bias = Dad[i][j].Bias
							else:
								Babby[i][j].Weights.append(Mom[i][j].Weights[k])
								Babby[i][j].Bias = Mom[i][j].Bias
							
							#chance to mutate
							if get_Random()<mu:
								if get_Random()>0.5:
									Babby[i][j].Weights[k]=Babby[i][j].Weights[k] - Babby[i][j].Weights[k]*muR
									if Babby[i][j].Weights[k]<0.:
											Babby[i][j].Weights[k]=0.
									Babby[i][j].Bias = Babby[i][j].Bias - Babby[i][j].Bias*muR
									if Babby[i][j].Bias<-1 :
										Babby[i][j].Bias=-1
								else:
									Babby[i][j].Weights[k]=Babby[i][j].Weights[k] + Babby[i][j].Weights[k]*muR
									if Babby[i][j].Weights[k]>1.:
										Babby[i][j].Weights[k]=1.
									Babby[i][j].Bias = Babby[i][j].Bias + Babby[i][j].Bias*muR
									if Babby[i][j].Bias>1 :
										Babby[i][j].Bias=1
							
	#outLayer
	Babby.append([])
	i=len(Babby)-1
	momi=len(Mom)-1
	dadi=len(Dad)-1
	#mom dad and babby have some j (num of output nodes) but not necessarily same i
	for j in range(len(Mom[momi])):
		Babby[len(Babby)-1].append(Node())
		
		for k in range(len(Babby[0])):
		
			if k> len(Mom[momi][j].Weights)-1 and k<= len(Dad[dadi][j].Weights)-1:
				Babby[i][j].Weights.append(Dad[dadi][j].Weights[k])
				Babby[i][j].Bias=Dad[dadi][j].Bias
			elif k <= len(Mom[momi][j].Weights)-1 and k > len(Dad[dadi][j].Weights)-1:
				Babby[i][j].Weights.append(Mom[momi][j].Weights[k])
				Babby[i][j].Bias=Mom[momi][j].Bias
			elif  k > len(Mom[momi][j].Weights)-1 and k > len(Dad[dadi][j].Weights)-1:
				Babby[i][j].Weights.append(get_Random())
				if get_Random()>0.5:
					Babby[i][j].Bias=-1*get_Random()
				else:
					Babby[i][j].Bias=get_Random()
			
			else:
				if get_Random() >0.5 :
					Babby[i][j].Weights.append(Dad[dadi][j].Weights[k])
					Babby[i][j].Bias = Dad[dadi][j].Bias
				else:
					Babby[i][j].Weights.append(Mom[momi][j].Weights[k])
					Babby[i][j].Bias = Mom[momi][j].Bias
			
				#chance to mutate
				if get_Random()<mu:
					if get_Random()>0.5:
						Babby[i][j].Weights[k]=Babby[i][j].Weights[k] - Babby[i][j].Weights[k]*muR
						if Babby[i][j].Weights[k]<0.:
								Babby[i][j].Weights[k]=0.
						Babby[i][j].Bias = Babby[i][j].Bias - Babby[i][j].Bias*muR
						if Babby[i][j].Bias<-1 :
							Babby[i][j].Bias=-1
					else:
						Babby[i][j].Weights[k]=Babby[i][j].Weights[k] + Babby[i][j].Weights[k]*muR
						if Babby[i][j].Weights[k]>1.:
							Babby[i][j].Weights[k]=1.
						Babby[i][j].Bias = Babby[i][j].Bias + Babby[i][j].Bias*muR
						if Babby[i][j].Bias>1 :
							Babby[i][j].Bias=1

	return Babby


##################################################################################		
def BreedBrains2(Mom,Dad,mu,muR):
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
	




def CheckBrain(inLen,outLen,Brain):
	
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

def Test_brain_breeding():
	Brains=[]
	for i in range(0,2):
		Brains.append(RandomBrain(random.randrange(2,3),random.randrange(2,3),20,1)) #Num of hidden layers, len of hidden layers
	while True:
		#Brains[1][1][1].Bias=0.5
	
		for Brain in Brains:
			#print "BRAIN!!!!!"
		
			try:
				for layer in Brain:
					printstr=''
					for t in range(0,len(layer)):
						try:
							printstr+=str(layer[t].Bias)+' '
						except:	
							printstr+=str(layer[t])+' '
					#print printstr
			except:
				#print Brain[len(Brain)-1].Bias
				continue
		

		NewBrains=[]
		for i in range(0,len(Brains)):
			print ''
		
			if i==len(Brains)-1:
			
				print "Mating Brains "+str(i)+" and "+str(0)
				printstr='Mom: '
				for layer in Brains[i]: printstr+=str(len(layer))+' '
				print printstr
				print len(Brains[i][len(Brains[i])-1][0].Weights)
			
				printstr='Dad: '
				for layer in Brains[0]: printstr+=str(len(layer))+' '
				print printstr
				print len(Brains[0][len(Brains[0])-1][0].Weights)
				NewBrains.append(BreedBrains(Brains[i],Brains[0]))
				printstr='Babby: '
				for layer in NewBrains[len(NewBrains)-1]: printstr+=str(len(layer))+' '
				print printstr
				print len(NewBrains[len(NewBrains)-1][len(NewBrains[len(NewBrains)-1])-1][0].Weights)
			
			else:
			
				print "Mating Brains "+str(i)+" and "+str(i+1)
				printstr='Mom: '
				for layer in Brains[i]: printstr+=str(len(layer))+' '
				print printstr
				print len(Brains[i][len(Brains[i])-1][0].Weights)
			
				printstr='Dad: '
				for layer in Brains[i+1]: printstr+=str(len(layer))+' '
				print printstr
				print len(Brains[i+1][len(Brains[i+1])-1][0].Weights)
				NewBrains.append(BreedBrains(Brains[i],Brains[i+1]))
				printstr='Babby: '
				for layer in NewBrains[len(NewBrains)-1]: printstr+=str(len(layer))+' '
				print printstr
				print len(NewBrains[len(NewBrains)-1][len(NewBrains[len(NewBrains)-1])-1][0].Weights)
				
	
		Brains=NewBrains
		raw_input("again?")		

		print "-----------------------------New Generation---------------------------------"


if __name__ == '__main__' :
	inputs =[get_Random() for i in range(30)]
	myBrain = RandomBrain(20,20,len(inputs),1)
	tone=clock()
	RunBrain(inputs,myBrain,3)
	dt=clock()-tone
	print myBrain[len(myBrain)-1][0].Value
	print str(int(round(dt*1000000,0)))+' us'
	
