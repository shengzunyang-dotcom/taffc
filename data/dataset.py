import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, BatchSampler



class MultimodalDataset(Dataset):
    def __init__(self, data, tokenizer=None):
        self.sentences = []
        self.input_ids = []
        self.attention_masks = []
        self.senti_labels = []
        self.segments = []
        self.visuals = []
        self.visual_length = []
        self.audios = []
        self.audio_length = []
        self.visual_masks = []
        self.audio_masks = []
        self.conv_labels = []
        
        self.text_masks = []

        for content in data:
            (text, visual, audio), senti_label, video_file = content
            if audio.shape[0] >= 100:
                flag_a = 0
                audio = audio[0: 100, :]
            else:
                flag_a = 1
                audio_len_p = audio.shape[0]
                
                audio = np.concatenate([audio,np.zeros((100-audio.shape[0],audio.shape[1]))],axis = 0)

            if visual.shape[0] >= 100:
                flag_v = 0
                visual = visual[0: 100, :]
            else:
                flag_v = 1
                visual_len_p = visual.shape[0]
                
                visual = np.concatenate([visual,np.zeros((100-visual.shape[0],visual.shape[1]))],axis = 0)
            if isinstance(senti_label, np.ndarray):
                #conv_label = np.clip(senti_label, a_min=-1., a_max=1.)
                #conv_label = np.round(conv_label) + 1
                conv_label = senti_label[0][0]
                # if conv_label < 0:
                #     conv_label = 0
                # elif conv_label > 0:
                #     conv_label = 1
                senti_label = senti_label[0][0]
            else:
                conv_label = senti_label

            # tokenized = tokenizer(text, add_special_tokens=True, return_tensors="pt", max_length=50, truncation=True, padding='max_length')
            tokenized = tokenizer(text, add_special_tokens=True, return_tensors="pt", max_length=100, truncation=True, padding='max_length')
            self.sentences.append(text)

            self.input_ids.append(tokenized['input_ids'].squeeze(0))
            self.attention_masks.append(tokenized['attention_mask'].squeeze(0))
            text_mask = tokenized['attention_mask'].squeeze(0)
            text_mask = ~text_mask.bool()
            self.text_masks.append(text_mask)


            senti_label = np.array([[senti_label]])
            senti_label = torch.tensor(senti_label.astype(np.float32)).cpu().detach()
            self.senti_labels.append(senti_label)

            conv_label = np.array([[conv_label]])
            conv_label = torch.tensor(conv_label.astype(np.float32)).cpu().detach()
            self.conv_labels.append(conv_label)


            
            self.segments.append(video_file)
            
            visual = torch.tensor(visual.astype(np.float32)).cpu().detach()
            self.visuals.append(visual)
            visual_len = visual.shape[0]
            self.visual_length.append(visual_len)

            audio = torch.tensor(audio.astype(np.float32)).cpu().detach()
            self.audios.append(audio)
            audio_len = audio.shape[0]
            self.audio_length.append(audio_len)

            if flag_v == 0:
                visual_mask = torch.arange(100) >= torch.tensor([100])
            else:
                # visual_mask = torch.ones(visual_len_p)
                # visual_mask = torch.cat([visual_mask,torch.zeros(100-visual_len_p)])
                visual_mask = torch.arange(100) >= torch.tensor([visual_len_p])
            self.visual_masks.append(visual_mask)
            if flag_a == 0:
                audio_mask = torch.arange(100) >= torch.tensor([100])
            else:
                # audio_mask = torch.ones(audio_len_p)
                # audio_mask = torch.cat([audio_mask,torch.zeros(100-audio_len_p)])
                audio_mask = torch.arange(100) >= torch.tensor([audio_len_p])
            self.audio_masks.append(audio_mask)



    def __len__(self):
        return len(self.senti_labels)
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        text_mask = self.text_masks[idx]
        senti_label = self.senti_labels[idx]
        conv_label = self.conv_labels[idx]

        segment = self.segments[idx]
        
        visual = self.visuals[idx]
        visual_len = self.visual_length[idx]
        
        audio = self.audios[idx]
        audio_len = self.audio_length[idx]
        
        visual_mask = self.visual_masks[idx]
        audio_mask = self.audio_masks[idx]

        

        return sentence, input_ids, attention_mask, text_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, senti_label, conv_label, segment



class MultimodalDGDataset(Dataset):
    def __init__(self, data, tokenizer=None):
        self.sentences = []
        self.input_ids = []
        self.attention_masks = []
        self.senti_labels = []
        self.segments = []
        self.visuals = []
        self.visual_length = []
        self.audios = []
        self.audio_length = []
        self.visual_masks = []
        self.audio_masks = []
        self.conv_labels = []

        for content in data:
            (text, visual, audio), senti_label, video_file = content
            if audio.shape[0] >= 100:
                audio = audio[0: 100, :]
            if visual.shape[0] >= 100:
                visual = visual[0: 100, :]
            if isinstance(senti_label, np.ndarray):
                senti_label = senti_label[0][0]
            else:
                senti_label = senti_label - 1
            conv_label = senti_label

            tokenized = tokenizer(text, add_special_tokens=True, return_tensors="pt", max_length=50, truncation=True, padding='max_length')
            self.sentences.append(text)

            self.input_ids.append(tokenized['input_ids'].squeeze(0))
            self.attention_masks.append(tokenized['attention_mask'].squeeze(0))


            senti_label = np.array([[senti_label]])
            senti_label = torch.tensor(senti_label.astype(np.float32)).cpu().detach()
            self.senti_labels.append(senti_label)

            conv_label = np.array([[conv_label]])
            conv_label = torch.tensor(conv_label.astype(np.float32)).cpu().detach()
            self.conv_labels.append(conv_label)


            
            self.segments.append(video_file)
            
            visual = torch.tensor(visual.astype(np.float32)).cpu().detach()
            self.visuals.append(visual)
            visual_len = visual.shape[0]
            self.visual_length.append(visual_len)

            audio = torch.tensor(audio.astype(np.float32)).cpu().detach()
            self.audios.append(audio)
            audio_len = audio.shape[0]
            self.audio_length.append(audio_len)

            visual_mask = torch.ones(visual_len)
            self.visual_masks.append(visual_mask)
            
            audio_mask = torch.ones(audio_len)
            self.audio_masks.append(audio_mask)


    def __len__(self):
        return len(self.senti_labels)
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        
        senti_label = self.senti_labels[idx]
        conv_label = self.conv_labels[idx]

        segment = self.segments[idx]
        
        visual = self.visuals[idx]
        visual_len = self.visual_length[idx]
        
        audio = self.audios[idx]
        audio_len = self.audio_length[idx]
        
        visual_mask = self.visual_masks[idx]
        audio_mask = self.audio_masks[idx]

        return sentence, input_ids, attention_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, senti_label, conv_label, segment



class MultimodalCLIPDataset(Dataset):
    def __init__(self, data, tokenizer=None):
        self.sentences = []
        self.input_ids = []
        self.attention_masks = []
        self.visuals = []
        self.labels = []
        self.segments = []
        self.visual_length = []
        for content in data:
            (words, visual), label_id, segment = content
            sentence = ' '.join(words)
            self.sentences.append(sentence)
            tokenized = tokenizer([sentence], return_tensors="pt", max_length=77, truncation=True, padding='max_length')
            self.input_ids.append(tokenized['input_ids'].squeeze(0))
            self.attention_masks.append(tokenized['attention_mask'].squeeze(0))
            visual = torch.tensor(visual.astype(np.float32)).cpu().detach()
            self.visuals.append(visual)
            visual_len = visual.shape[0]
            # visual_len = torch.tensor(visual_len).cpu().detach()
            self.visual_length.append(visual_len)
            label_id = torch.tensor(label_id.astype(np.float32)).cpu().detach()
            self.labels.append(label_id)
            self.segments.append(segment)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        visual = self.visuals[idx]
        label_id = self.labels[idx]
        segment = self.segments[idx]
        visual_len = self.visual_length[idx]
        return sentence, input_ids, attention_mask, visual, visual_len, label_id, segment


class MultimodalImagebindDataset(Dataset):
    def __init__(self, data, tokenizer=None):
        self.texts = []
        self.visuals = []
        self.audios = []
        self.labels = []
        self.segments = []
        self.visual_length = []
        for content in data:
            (text_features, vision_features, audio_features), label_id, segment = content
            self.texts.append(torch.tensor(text_features.astype(np.float32)).cpu().detach())
            self.audios.append(torch.tensor(audio_features.astype(np.float32)).cpu().detach())
            self.visuals.append(torch.tensor(vision_features.astype(np.float32)).cpu().detach())
            self.visual_length.append(vision_features.shape[0])
            label_id = torch.tensor(label_id.astype(np.float32)).cpu().detach()
            self.labels.append(label_id)
            self.segments.append(segment)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        text = self.texts[idx]
        visual = self.visuals[idx]
        audio = self.audios[idx]
        label_id = self.labels[idx]
        segment = self.segments[idx]
        visual_len = self.visual_length[idx]
        return text, visual, audio, visual_len, label_id, segment
    

class MultimodalMeldMultiLabelDataset(Dataset):
    def __init__(self, data, tokenizer=None):
        self.sentences = []
        self.input_ids = []
        self.attention_masks = []
        self.senti_labels = []
        self.segments = []
        self.emo_labels = []

        self.visuals = []
        self.visual_length = []

        for content in data:
            (text, visual), senti_label, emo_label, video_file = content

            tokenized = tokenizer(text, add_special_tokens=True, return_tensors="pt", max_length=50, truncation=True, padding='max_length')
            self.sentences.append(text)
            self.input_ids.append(tokenized['input_ids'].squeeze(0))
            self.attention_masks.append(tokenized['attention_mask'].squeeze(0))
            
            senti_label = np.array([[senti_label]])
            senti_label = torch.tensor(senti_label.astype(np.float32)).cpu().detach()
            self.senti_labels.append(senti_label)

            emo_label = np.array([[emo_label]])
            emo_label = torch.tensor(emo_label.astype(np.float32)).cpu().detach()
            self.emo_labels.append(emo_label)


            self.segments.append(video_file)

            visual = torch.tensor(visual.astype(np.float32)).cpu().detach()
            self.visuals.append(visual)
            visual_len = visual.shape[0]
            self.visual_length.append(visual_len)

    def __len__(self):
        return len(self.senti_labels)
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        senti_label = self.senti_labels[idx]
        segment = self.segments[idx]
        emo_label = self.emo_labels[idx]
        visual = self.visuals[idx]
        visual_len = self.visual_length[idx]
        return sentence, input_ids, attention_mask, visual, visual_len, emo_label, senti_label, segment

class MultimodalMeldDataset(Dataset):
    def __init__(self, data, tokenizer=None):
        self.sentences = []
        self.input_ids = []
        self.attention_masks = []
        self.senti_labels = []
        self.segments = []
        self.visuals = []
        self.visual_length = []

        for content in data:
            (text, visual), senti_label, video_file = content
            if isinstance(senti_label, np.ndarray):
                senti_label = np.clip(senti_label, a_min=-1., a_max=1.)
                senti_label = np.round(senti_label) + 1
                senti_label = senti_label[0][0]
            tokenized = tokenizer(text, add_special_tokens=True, return_tensors="pt", max_length=50, truncation=True, padding='max_length')
            self.sentences.append(text)
            self.input_ids.append(tokenized['input_ids'].squeeze(0))
            self.attention_masks.append(tokenized['attention_mask'].squeeze(0))
            senti_label = np.array([[senti_label]])
            senti_label = torch.tensor(senti_label.astype(np.float32)).cpu().detach()
            self.senti_labels.append(senti_label)
            self.segments.append(video_file)
            visual = torch.tensor(visual.astype(np.float32)).cpu().detach()
            self.visuals.append(visual)
            visual_len = visual.shape[0]
            self.visual_length.append(visual_len)

    def __len__(self):
        return len(self.senti_labels)
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        senti_label = self.senti_labels[idx]
        segment = self.segments[idx]
        visual = self.visuals[idx]
        visual_len = self.visual_length[idx]
        return sentence, input_ids, attention_mask, visual, visual_len, senti_label, senti_label, segment

class MultimodalMOSEIDataset(Dataset):
    def __init__(self, data, tokenizer=None):
        self.sentences = []
        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        self.senti_labels = []
        self.segments = []
        self.visual_length = []

        self.visuals = []
        self.visual_length = []

        for content in data:
            (words, visual), label_id, segment = content
            # sentence = ' '.join(words)
            sentence = words
            self.sentences.append(sentence)
            tokenized = tokenizer([sentence], return_tensors="pt", max_length=50, truncation=True, padding='max_length')
            # add_special_tokens=True, 
            self.input_ids.append(tokenized['input_ids'].squeeze(0))
            self.attention_masks.append(tokenized['attention_mask'].squeeze(0))            
            senti_label = np.clip(label_id, a_min=-1., a_max=1.)
            senti_label = np.round(senti_label) + 1
            senti_label = torch.tensor(senti_label.astype(np.float32)).cpu().detach()
            label_id = torch.tensor(label_id.astype(np.float32)).cpu().detach()
            self.senti_labels.append(senti_label)
            self.labels.append(label_id)
            self.segments.append(segment)

            visual = torch.tensor(visual.astype(np.float32)).cpu().detach()
            self.visuals.append(visual)
            visual_len = visual.shape[0]
            self.visual_length.append(visual_len)

    def __len__(self):
        return len(self.senti_labels)
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        senti_label = self.senti_labels[idx]
        label_id = self.labels[idx]
        segment = self.segments[idx]
        visual = self.visuals[idx]
        visual_len = self.visual_length[idx]
        return sentence, input_ids, attention_mask, visual, visual_len, label_id, senti_label, segment


class MultimodalMOSIDataset(Dataset):
    def __init__(self, data, tokenizer=None):
        self.sentences = []
        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        self.senti_labels = []
        self.segments = []

        self.visuals = []
        self.visual_length = []


        for content in data:
            (words, visual), label_id, segment = content
            # sentence = ' '.join(words)
            sentence = words
            self.sentences.append(sentence)
            tokenized = tokenizer([sentence], add_special_tokens=True,  return_tensors="pt", max_length=50, truncation=True, padding='max_length')
            self.input_ids.append(tokenized['input_ids'].squeeze(0))
            self.attention_masks.append(tokenized['attention_mask'].squeeze(0))

            senti_label = np.clip(label_id, a_min=-1., a_max=1.)
            senti_label = np.round(senti_label) + 1
            senti_label = torch.tensor(senti_label.astype(np.float32)).cpu().detach()
            label_id = torch.tensor(label_id.astype(np.float32)).cpu().detach()
            self.senti_labels.append(senti_label)
            self.labels.append(label_id)
            self.segments.append(segment)

            visual = torch.tensor(visual.astype(np.float32)).cpu().detach()
            self.visuals.append(visual)
            visual_len = visual.shape[0]
            self.visual_length.append(visual_len)

    def __len__(self):
        return len(self.senti_labels)
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        senti_label = self.senti_labels[idx]
        label_id = self.labels[idx]
        segment = self.segments[idx]

        visual = self.visuals[idx]
        visual_len = self.visual_length[idx]
        return sentence, input_ids, attention_mask, visual, visual_len, label_id, senti_label, segment

# dataset = MultimodalImagebindDataset()