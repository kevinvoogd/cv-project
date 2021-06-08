import pickle
import numpy
import matplotlib.pyplot as plt


training_loss = []
testing_loss = []
with open('Training_loss.pkl','rb') as f:
    while True:
        try:
            training_loss.append(pickle.load(f))
        except EOFError:
            break

with open('Test_loss.pkl','rb') as f:
    while True:
        try:
            testing_loss.append(pickle.load(f))
        except EOFError:
            break



train_loss = numpy.array(training_loss) # changing to array
train_loss= train_loss.reshape((-1,1)) #making a single column
div = int(train_loss.shape[0]/100) # getting the last iteration divisible by 100
train_loss = train_loss[:div*100]
train_loss = train_loss.reshape((-1,10)).mean(axis=1) # takin mean of 100 iterations

test_loss = numpy.array(testing_loss)
test_loss = test_loss.reshape((-1,1))
div = int(test_loss.shape[0]/100)
test_loss = test_loss[:div*100]
test_loss = test_loss.reshape((-1,10)).mean(axis=1)


plt.figure(1)
plt.title("Training Loss")
plt.ylabel("Loss")
plt.xlabel("Per Iteration (Batch Size 15)")
plt.plot(train_loss)

plt.figure(2)
plt.title("Testing Loss")
plt.ylabel("Loss")
plt.xlabel("Per Iteration (Batch Size 15)")
plt.plot(test_loss)

plt.show()
