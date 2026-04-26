import csv
import os
import pickle
def get_data():
    train_csv = '/disk3/multimodal-dataset/meld/MELD.Raw/train_sent_emo.csv'
    dev_csv = '/disk3/multimodal-dataset/meld/MELD.Raw/dev_sent_emo.csv'
    test_csv = '/disk3/multimodal-dataset/meld/MELD.Raw/test_sent_emo.csv'

    emo_label_dict = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6}
    sent_label_dict = {'negative': 0, 'neutral': 1, 'positive': 2}


    data = {}

    train_set = []
    train_csv_reader = csv.reader(open(train_csv))
    count = 0
    for line in train_csv_reader:
        if count == 0:
            count += 1
            continue
        # print(line)
        count += 1
        sr_no = line[0]
        text = line[1]
        speaker = line[2]
        emotion = line[3]
        sentiment = line[4]
        dialogue = line[5]
        utterance = line[6]
        
        video_file = f'dia{dialogue}_utt{utterance}'
        emo_label = emo_label_dict[emotion]
        sent_label = sent_label_dict[sentiment]

        # print(text, emotion, sentiment, dialogue, utterance, sent_label, emo_label, video_file)
        temp = (text, video_file, sent_label, emo_label)
        train_set.append(temp)


    dev_set = []
    dev_csv_reader = csv.reader(open(dev_csv))
    count = 0
    for line in dev_csv_reader:
        if count == 0:
            count += 1
            continue
        #print(line)
        count += 1
        sr_no = line[0]
        text = line[1]
        speaker = line[2]
        emotion = line[3]
        sentiment = line[4]
        dialogue = line[5]
        utterance = line[6]
        
        video_file = f'dia{dialogue}_utt{utterance}'
        emo_label = emo_label_dict[emotion]
        sent_label = sent_label_dict[sentiment]

        #print(text, emotion, sentiment, dialogue, utterance, sent_label, emo_label, video_file)
        temp = (text, video_file, sent_label, emo_label)
        dev_set.append(temp)
    
    test_set = []
    test_csv_reader = csv.reader(open(test_csv))
    count = 0
    for line in test_csv_reader:
        if count == 0:
            count += 1
            continue
        #print(line)
        count += 1
        sr_no = line[0]
        text = line[1]
        speaker = line[2]
        emotion = line[3]
        sentiment = line[4]
        dialogue = line[5]
        utterance = line[6]
        
        video_file = f'dia{dialogue}_utt{utterance}'
        emo_label = emo_label_dict[emotion]
        sent_label = sent_label_dict[sentiment]

        #print(text, emotion, sentiment, dialogue, utterance, sent_label, emo_label, video_file)
        temp = (text, video_file, sent_label, emo_label)
        test_set.append(temp)

    data = {'train': train_set, 'dev': dev_set, 'test': test_set}
    return data






data = get_data()

with open('/home/zhaoxianbing/CMU/llm-features/meld/meld_meta.pkl', 'wb') as f:
    pickle.dump(data, f)