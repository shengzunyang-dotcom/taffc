import pickle

mosei_hubert_path = '/disk4/CMU-RAW/MOSEI/hubert_all.pkl'
with open(mosei_hubert_path, "rb") as handle:
    mosei_hubert = pickle.load(handle)
mosei_prefix = '/disk4/CMU-RAW/MOSEI/Raw/Audio/Segmented/'
mosei_suffix = '.wav'



mosei_data_path = '/disk4/cmu-features/mosei/mosei_vgg.pkl'
with open(mosei_data_path, "rb") as handle:
    mosei_data = pickle.load(handle)



mosei_train = []
mosei_dev = []
mosei_test = []

for content in mosei_data['train']:
    (words, visual), label_id, segment = content
    key = mosei_prefix + segment + mosei_suffix
    audio = mosei_hubert[key][0]
    temp = (words, visual, audio), label_id, segment
    mosei_train.append(temp)

for content in mosei_data['dev']:
    (words, visual), label_id, segment = content
    key = mosei_prefix + segment + mosei_suffix
    audio = mosei_hubert[key][0]
    temp = (words, visual, audio), label_id, segment
    mosei_dev.append(temp)

for content in mosei_data['test']:
    (words, visual), label_id, segment = content
    key = mosei_prefix + segment + mosei_suffix
    audio = mosei_hubert[key][0]
    temp = (words, visual, audio), label_id, segment
    mosei_test.append(temp)

mosei_vgg_hubert = {"train": mosei_train, "dev": mosei_dev, "test": mosei_test}

with open('/disk4/cmu-features/merge/mosei_vgg_hubert.pkl', 'wb') as f:
    pickle.dump(mosei_vgg_hubert, f)



