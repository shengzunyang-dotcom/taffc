import pickle

mosi_hubert_path = '/disk4/CMU-RAW/MOSI/hubert_all.pkl'
with open(mosi_hubert_path, "rb") as handle:
    mosi_hubert = pickle.load(handle)
mosi_prefix = '/disk4/CMU-RAW/MOSI/Raw/Audio/Segmented/'
mosi_suffix = '.wav'
mosi_data_path = '/disk4/cmu-features/mosi/mosi_vgg.pkl'
with open(mosi_data_path, "rb") as handle:
    mosi_data = pickle.load(handle)

mosi_train = []
mosi_dev = []
mosi_test = []

for content in mosi_data['train']:
    (words, visual), label_id, segment = content
    key = mosi_prefix + segment + mosi_suffix
    audio = mosi_hubert[key][0]
    temp = (words, visual, audio), label_id, segment
    mosi_train.append(temp)

for content in mosi_data['dev']:
    (words, visual), label_id, segment = content
    key = mosi_prefix + segment + mosi_suffix
    audio = mosi_hubert[key][0]
    temp = (words, visual, audio), label_id, segment
    mosi_dev.append(temp)

for content in mosi_data['test']:
    (words, visual), label_id, segment = content
    key = mosi_prefix + segment + mosi_suffix
    audio = mosi_hubert[key][0]
    temp = (words, visual, audio), label_id, segment
    mosi_test.append(temp)

mosi_vgg_hubert = {"train": mosi_train, "dev": mosi_dev, "test": mosi_test}

with open('/disk4/cmu-features/merge/mosi_vgg_hubert.pkl', 'wb') as f:
    pickle.dump(mosi_vgg_hubert, f)

exit(0)



mosei_data_path = '/disk4/cmu-features/mosei/mosei_vgg.pkl'
with open(mosei_data_path, "rb") as handle:
    mosei_data = pickle.load(handle)


meld_data_path = '/disk4/cmu-features/meld/meld_vgg.pkl'
with open(meld_data_path, "rb") as handle:
    meld_data = pickle.load(handle)





# mosi 1283 229 686
# mosei 16097 1858 4591
# meld 9965 1104 2608



with open('/disk4/cmu-features/merge/meld_vgg_hubert.pkl', 'wb') as f:
    pickle.dump(meld, f)

with open('/disk4/cmu-features/merge/mosei_vgg_hubert.pkl', 'wb') as f:
    pickle.dump(mosei, f)
