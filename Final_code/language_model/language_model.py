import numpy as np
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
import re
import sys
#print(common_texts)
print(len(sys.argv))
if(len(sys.argv)<4):
	print("4 arguments are neaded\n")
	exit()
data=""
dp=[]
with open(sys.argv[1],'r') as training_data:
	for line in training_data:
		line=re.sub('[\n]',"",line)
		dp.append(line)
		data+=line
		data+= " "


# integer encode text

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
#print('r')
encoded = tokenizer.texts_to_sequences(data)[0]

print(tokenizer.texts_to_sequences(data)[0])
# determine the vocabulary size
indexer= tokenizer.word_index

vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

# create word -> word sequences
sequences = list()
for row in dp:
	tokens=row.split(" ")
	if(len(tokens)<3):
		continue
	for i in range(2,len(tokens)):
		sequence = [indexer[tokens[i-2]],indexer[tokens[i]],indexer[tokens[i-1]] ]
		#print(sequence)
		sequences.append(sequence)

print('Total Sequences: %d' % len(sequences))
max_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
print(max_length)

fl=open(sys.argv[2],'w')
for key,value in indexer.iteritems():
	fl.write(str(key)+ ': ' + str(value)+'\n')
# split into X and y elements
fl.close()
sequences = array(sequences)
#exit()
i=0
model = Sequential()
model.add(Embedding(vocab_size, 128 ,input_length=max_length-1))
model.add(LSTM(120))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
leng=len(sequences)
print len(sequences)

i=0
for _ in range(20):
	i=0
	while i < len(sequences):
		if(i+1000<len(sequences)):
			X, y = sequences[i:i+1000,:-1],sequences[i:i+1000,-1]
		else:
			X, y = sequences[i:,:-1],sequences[i:,-1]
		print i
		y = to_categorical(y, num_classes=vocab_size)
		model.fit(X, y, epochs=1, batch_size=100, verbose=1)
		i+=1000
	model.save(sys.argv[3])



'''
while i < leng:
	X, y = sequences[i,0],sequences[i,1]

	# one hot encode outputs
	# define model

	# fit network
	y = to_categorical(y, num_classes=vocab_size)
	print i
	print X
	print y
	model.fit(X, y)

	i+=1
'''
# evaluate

#print(generate_seq(model, tokenizer, 'Jack', 6))
