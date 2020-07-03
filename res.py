import torch
from net import CNNModel
import pickle
import numpy as np
import matplotlib.pyplot as plt

model = torch.load('trainednet')
test_data = pickle.load( open( "test_loader.p", "rb" ) )
loss = pickle.load( open( "loss.p", "rb" ))
accuracy = pickle.load( open( "acc_list.p", "rb" ))

nb_classes = 11
confusion_matrix = torch.zeros(nb_classes, nb_classes)

with torch.no_grad():
    for i,(inputs, classes) in enumerate(test_data):
        print(f"batch {i}")
        inputs = inputs.view(50,4,40,32,32)
        outputs = model(inputs)
        _, preds = torch.max(outputs,1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()]+=1

conf2 = confusion_matrix.diag()/confusion_matrix.sum(1)
print(confusion_matrix)
print(conf2)

pickle.dump(confusion_matrix, open( "conf1.p", "wb" ) )
pickle.dump(conf2, open( "conf2.p", "wb" ) )
