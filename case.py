import pickle


# path = "/data/zhaoxianbing/CMU/process_data/mosi_unalign.pkl"
path = "/data/zhaoxianbing/CMU/datasets/mosi_images.pkl"
with open(path, "rb") as handle:
        mosi_data = pickle.load(handle)

# for key, value in mosi_data.items():
#     # print(f"{key}: {value}")
#     print(f"{key}")

# print(mosi_data['train'][0][0][2].shape)

print(mosi_data['LSi-o-IrDMs_11'])


