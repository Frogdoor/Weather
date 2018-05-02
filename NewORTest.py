import pickle
from BRAIN_BUILDER import *


#---Variables------------------------------------
NumLayers	= 3	#Number of hidden layers
NumNodes 	= 4	#Number of nodes per hidden layer
NumInputs	= 1 #Number of Input nodes
NumOutputs	= 1	#Number of Ouput nodes

cntr = 0 #Training Itteration counter

LearnRate = .25 #Backpropation learning rate

tol=0.1 #tolerance on answer

nCores = 1 #how many cores belong to the brain
#-----------------------------------------------


#Create the Brain
theBrain = RandomBrain(NumLayers,NumNodes,NumInputs,NumOutputs)

#start data logging
f = open("LearningCurve.dat", 'a')
f.write("\n\n")


def randomCoin(): # flip a coin return 1 or 0
	x= 1 if random.random() < 0.5 else 0
	return x
	
	
def TraningFunction():
	
	Ins=[]
	for i in range(NumInputs):
		Ins.append(random.random())
	
	Ans =  pow(Ins[0],2)

#	And/OR function
#	Ins=[]
#	for i in range(NumInputs) :
#		Ins.append(randomCoin())
#		
#	Ans = all(Ins)
#	
	return Ins,Ans


#Training loop
while True:
	
	#Get some traning Data
	inputs,answer = TraningFunction()

	#Run the Brain
	RunBrain(inputs,theBrain,nCores)
	
	guess = round(theBrain[len(theBrain)-1][0].Value,2)
	
	
	#Train the brain
	BackProp(theBrain,[answer],LearnRate) #Brain, Key, rate
	cntr += 1


	#Test the brain
	test=[]
	for i in range(100):
		inputs,answer = TraningFunction()
		
		RunBrain(inputs,theBrain,nCores)
		guess = round(theBrain[len(theBrain)-1][0].Value,2)
		
		test.append((abs(guess-answer)<tol))
		
	f.write(str(cntr)+" " + str(sum(test)) + "\n")
		
	if cntr%10 == 0 : print "Gen " + str(cntr) +" - " "Score: " + str(sum(test)) + "/" + str(len(test)) 
	if all(test): break
	
print "Brain Trained! It took "+str(cntr)+" training sessions."


f.close()

while True:

	loop= raw_input("Test again? :")
	
	if loop != 'y' and loop != 'Y': break

	test=[]
	for i in range(20):
		inputs,answer = TraningFunction()
		
		RunBrain(inputs,theBrain,nCores)
		guess = round(theBrain[len(theBrain)-1][0].Value,2)
		
		test.append((abs(guess-answer)<tol))
		print "Wanted: " +str(answer)+"\tGot: "+str(guess) 
	
	print str(sum(test)*100/len(test)) + "% Success Rate"

	
	
