import pickle
from BRAIN_BUILDER import *

def randomCoin(): # flip a coin return 1 or 0
	x= 1 if random.random() < 0.5 else 0
	return x
	

#RandomBrain(NumLayers,numNodes,numInputs,numOutputs)
theBrain = RandomBrain(1,2,2,1)

inputs=[None]*2

tol=0.15 #tolerance on answer

cntr=0
cntr2=0

while True:
	#fill the inputs with random 1s and 0s
	for i in range(len(inputs)) :
		inputs[i] = randomCoin()

	#We want to train for OR condition
	answer = int(any(inputs))

	#Run the Brain
	RunBrain(inputs,theBrain,1)
	
	guess = round(theBrain[len(theBrain)-1][0].Value,2)
	
	print "Wanted: " +str(answer)+"\tGot: "+str(guess)
	
	#Train the brain
	BackProp(theBrain,[answer],1.)
	cntr += 1

	test=[]
	for i in range(20):
		q = [1,0] if i%2==0 else [0,0]
		a = RunBrain(q,theBrain,1)
		test.append((abs(guess-answer)<tol))
	
	if all(test): break
	
print "Brain Trained! It took "+str(cntr)+" training sessions."




	
	
