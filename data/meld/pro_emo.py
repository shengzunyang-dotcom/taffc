import pickle

with open('/home/zhaoxianbing/CMU/llm-features/meld/meld_meta.pkl', "rb") as f1:
    meta_data = pickle.load(f1)


with open('/home/zhaoxianbing/CMU/llm-features/meld/meld_clip.pkl', 'rb') as f2:
    data = pickle.load(f2)


label_dict = {}
print(len(meta_data['train']), len(meta_data['dev']), len(meta_data['test']), len(meta_data['train']) + len(meta_data['dev']) + len(meta_data['test']))
for idx in range(len(meta_data['train'])):
    (text, video_file, sent_label, emo_label) = meta_data['train'][idx]
    label_dict['train' + video_file] = [sent_label, emo_label]

for idx in range(len(meta_data['dev'])):
    (text, video_file, sent_label, emo_label) = meta_data['dev'][idx]
    label_dict['dev' + video_file] = [sent_label, emo_label]

for idx in range(len(meta_data['test'])):
    (text, video_file, sent_label, emo_label) = meta_data['test'][idx]
    label_dict['test' + video_file] = [sent_label, emo_label]

print(len(label_dict.keys()))



train, dev, test = [], [], []

for idx in range(len(data['train'])):
    (text, image_embeds), sent_label, video_file = data['train'][idx]
    [sent_label1, emo_label1] = label_dict['train' + video_file]
    
    temp = (text, image_embeds), sent_label, emo_label1, video_file
    train.append(temp)
    # print(sent_label, sent_label1, emo_label1)
    if sent_label1 != sent_label:
        print('error!!!')
        exit(0)

for idx in range(len(data['dev'])):
    (text, image_embeds), sent_label, video_file = data['dev'][idx]
    [sent_label1, emo_label1] = label_dict['dev' + video_file]
    
    temp = (text, image_embeds), sent_label, emo_label1, video_file
    dev.append(temp)
    
    if sent_label1 != sent_label:
        print('error!!!')
        exit(0)
for idx in range(len(data['test'])):
    (text, image_embeds), sent_label, video_file = data['test'][idx]
    [sent_label1, emo_label1] = label_dict['test' + video_file]
    
    temp = (text, image_embeds), sent_label, emo_label1, video_file
    test.append(temp)

    if sent_label1 != sent_label:
        print('error!!!')
        exit(0)

all_data = {"train": train, "dev": dev, "test": test}
with open('/home/zhaoxianbing/CMU/llm-features/meld/meld_clip_senti_emo.pkl', 'wb') as f:
    pickle.dump(all_data, f)










