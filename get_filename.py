import json
train_json='/tmp/xinyuw3/imsitu/train.json'
dev_json='/tmp/xinyuw3/imsitu/dev.json'
test_json='/tmp/xinyuw3/imsitu/test.json'



with open(train_json, 'r') as j:
    train_data = json.load(j)

with open(dev_json, 'r') as j:
    dev_data = json.load(j)

with open(test_json, 'r') as j:
    test_data = json.load(j)

train_names=[]
val_names=[]
test_names=[]
for key in train_data:
    if key != 'crouching_150.jpg' and key!= 'steering_72.jpg':
        train_names.append(key)

for key in dev_data:
    val_names.append(key)

for key in test_data:
    if key != 'pouting_177.jpg':
        test_names.append(key)

f = open('train_names.txt','w')
for name in train_names:
    f.write(name+'\n')

f = open('val_names.txt','w')
for name in val_names:
    f.write(name+'\n')

f = open('test_names.txt','w')
for name in test_names:
    f.write(name+'\n')




