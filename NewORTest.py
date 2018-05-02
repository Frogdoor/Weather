import pickle
from BRAIN_BUILDER import *


def randomCoin(): # flip a coin return 1 or 0
	x= 1 if random.random() < 0.5 else 0
	return x
	
	
def GenTrainSet(NumInputs):
	Ins=[]
	for i in range(NumInputs) :
		Ins.append(randomCoin())
		
	Ans = any(Ins)
	
	return Ins,Ans


#RandomBrain(NumLayers,numNodes,numInputs,numOutputs)
theBrain = RandomBrain(1,4,2,1)



tol=0.15 #tolerance on answer

cntr=0
cntr2=0

while True:


	inputs,answer = GenTrainSet(2)

	#Run the Brain
	RunBrain(inputs,theBrain,1)
	
	guess = round(theBrain[len(theBrain)-1][0].Value,2)
	
	#print "Wanted: " +str(answer)+"\tGot: "+str(guess)
	
	#Train the brain
	BackProp(theBrain,[answer],1.)
	cntr += 1


	#print "------TEST!!!!-------"
	test=[]
	for i in range(100):
		inputs,answer = GenTrainSet(2)
		
		RunBrain(inputs,theBrain,1)
		guess = round(theBrain[len(theBrain)-1][0].Value,2)
		
		
		#print guess,answer
		test.append((abs(guess-answer)<tol))
		
	if cntr%10 == 0 : print "Gen " + str(cntr) +" - " "Score: " + str(sum(test)) + "/" + str(len(test)) 
	if all(test): break
	
print "Brain Trained! It took "+str(cntr)+" training sessions."


while True:

	loop= raw_input("Test again? :")
	
	if loop != 'y' and loop != 'Y': break

	test=[]
	for i in range(20):
		inputs,answer = GenTrainSet(2)
		
		RunBrain(inputs,theBrain,1)
		guess = round(theBrain[len(theBrain)-1][0].Value,2)
		
		test.append((abs(guess-answer)<tol))
		print "Wanted: " +str(answer)+"\tGot: "+str(guess) 
	
	print str(sum(test)*100/len(test)) + "% Success Rate"

	
	
