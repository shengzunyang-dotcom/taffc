
import pickle
mosi_data_path = '/disk4/cmu-features/mosi/mosi_vgg.pkl'
with open(mosi_data_path, "rb") as handle:
    mosi_data = pickle.load(handle)
mosi_train_data = mosi_data['train']
mosi_dev_data = mosi_data['dev']
mosi_test_data = mosi_data['test']
for content in mosi_train_data:
    (words, visual), label_id, segment = content
    print(type(label_id), label_id)
    break
print(len(mosi_train_data), len(mosi_dev_data), len(mosi_test_data))


mosei_data_path = '/disk4/cmu-features/mosei/mosei_vgg.pkl'
with open(mosei_data_path, "rb") as handle:
    mosei_data = pickle.load(handle)
mosei_train_data = mosei_data['train']
mosei_dev_data = mosei_data['dev']
mosei_test_data = mosei_data['test']
for content in mosei_train_data:
    (words, visual), label_id, segment = content
    print(type(label_id), label_id)
    break
print(len(mosei_train_data), len(mosei_dev_data), len(mosei_test_data))


meld_data_path = '/disk4/cmu-features/meld/meld_vgg.pkl'
with open(meld_data_path, "rb") as handle:
    meld_data = pickle.load(handle)
meld_train_data = meld_data['train']
meld_dev_data = meld_data['dev']
meld_test_data = meld_data['test']
for content in meld_train_data:
    (words, visual), label_id, segment = content
    print(type(label_id), label_id)
    break
print(len(meld_train_data), len(meld_dev_data), len(meld_test_data))


mosi_mosei_train = mosi_train_data + mosei_train_data
mosi_mosei_dev = mosi_dev_data + mosei_dev_data


mosi_meld_train = mosi_train_data + meld_train_data
mosi_meld_dev = mosi_dev_data + meld_dev_data


mosei_meld_train = mosei_train_data + meld_train_data
mosei_meld_dev = mosei_dev_data + meld_dev_data


mosi_test = mosi_test_data
mosei_test = mosei_test_data
meld_test = meld_test_data


mosi_mosei = {'train': mosi_mosei_train, 'dev': mosi_mosei_dev, 'test_mosi': mosi_test, 'test_mosei': mosei_test, 'test_meld': meld_test}
mosi_meld = {'train': mosi_meld_train, 'dev': mosi_meld_dev, 'test_mosi': mosi_test, 'test_mosei': mosei_test, 'test_meld': meld_test}
mosei_meld = {'train': mosei_meld_train, 'dev': mosei_meld_dev, 'test_mosi': mosi_test, 'test_mosei': mosei_test, 'test_meld': meld_test}
# mosi 1283 229 686
# mosei 16097 1858 4591
# meld 9965 1104 2608

with open('/disk4/cmu-features/merge/mosi_mosei_vgg.pkl', 'wb') as f:
    pickle.dump(mosi_mosei, f)

with open('/disk4/cmu-features/merge/mosi_meld_vgg.pkl', 'wb') as f:
    pickle.dump(mosi_meld, f)

with open('/disk4/cmu-features/merge/mosei_meld_vgg.pkl', 'wb') as f:
    pickle.dump(mosei_meld, f)
