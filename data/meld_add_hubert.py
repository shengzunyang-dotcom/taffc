import pickle

meld_train_hubert_path = '/disk4/CMU-RAW/MELD/hubert_train.pkl'
with open(meld_train_hubert_path, "rb") as handle:
    meld_train_hubert = pickle.load(handle)

meld_dev_hubert_path = '/disk4/CMU-RAW/MELD/hubert_dev.pkl'
with open(meld_dev_hubert_path, "rb") as handle:
    meld_dev_hubert = pickle.load(handle)

meld_test_hubert_path = '/disk4/CMU-RAW/MELD/hubert_test.pkl'
with open(meld_test_hubert_path, "rb") as handle:
    meld_test_hubert = pickle.load(handle)


meld_train_prefix = '/disk4/CMU-RAW/MELD/MELD/train_splits_audio/'
meld_dev_prefix = '/disk4/CMU-RAW/MELD/MELD/dev_splits_audio/'
meld_test_prefix = '/disk4/CMU-RAW/MELD/MELD/test_splits_audio/'
meld_suffix = '.wav'



meld_data_path = '/disk4/cmu-features/meld/meld_vgg.pkl'
with open(meld_data_path, "rb") as handle:
    meld_data = pickle.load(handle)

meld_train = []
meld_dev = []
meld_test = []

for content in meld_data['train']:
    (words, visual), label_id, segment = content
    key = meld_train_prefix + segment + meld_suffix
    audio = meld_train_hubert[key][0]
    temp = (words, visual, audio), label_id, segment
    meld_train.append(temp)



for content in meld_data['dev']:
    (words, visual), label_id, segment = content
    key = meld_dev_prefix + segment + meld_suffix
    audio = meld_dev_hubert[key][0]
    temp = (words, visual, audio), label_id, segment
    meld_dev.append(temp)

for content in meld_data['test']:
    (words, visual), label_id, segment = content
    key = meld_test_prefix + segment + meld_suffix
    if key not in meld_test_hubert.keys():
        continue
    audio = meld_test_hubert[key][0]
    temp = (words, visual, audio), label_id, segment
    meld_test.append(temp)

meld_vgg_hubert = {"train": meld_train, "dev": meld_dev, "test": meld_test}

with open('/disk4/cmu-features/merge/meld_vgg_hubert.pkl', 'wb') as f:
    pickle.dump(meld_vgg_hubert, f)






# mosi 1283 229 686
# mosei 16097 1858 4591
# meld 9965 1104 2608



