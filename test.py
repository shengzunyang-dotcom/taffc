# import torch
# import torch.nn.functional as F
import pickle as pkl
# import torch.nn as nn
import math
# from torch.nn import functional as F, init
import numpy as np
import torch,gc
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  
os.environ["TORCH_USE_CUDA_DSA"] = "1" 
# modality_prompt = nn.Parameter(
#             torch.empty((3,5,5))
#         )
# for i in modality_prompt:
#     print(i)
#     init.kaiming_uniform_(i, a=math.sqrt(5))
#     print(i)
# for i in np.arange(0,86):
#     path = f"/data/yangshengzun/knowledge-injection/visualization/visualization-1-{i}.pkl"
#     with open(path, "rb") as handle:
#         mosi_data = pkl.load(handle)
#         for j in mosi_data['text']:
#             print(j)
#         print(mosi_data['sim_vector'][0])
#         print(mosi_data['sim_vector'][1])
#         print(mosi_data['sim_vector'][2])
#         print(mosi_data['label'])
#         print(mosi_data['output'])
#         print(mosi_data['id'])
for i in np.arange(1,162)[:30]:
    path = f"/data/yangshengzun/knowledge-injection/visualize/visualization-1-{i}.pkl"
    # path = f"/data/yangshengzun/knowledge-injection/visualize/visualization-1-152.pkl"
    with open(path, "rb") as handle:
        mosi_data = pkl.load(handle)
        for j in mosi_data['text']:
            print(j)
        print(mosi_data['sim_vector'][0][0],mosi_data['sim_vector'][1][0])
        print(mosi_data['sim_vector'][0][1],mosi_data['sim_vector'][1][1])
        print(mosi_data['sim_vector'][0][2],mosi_data['sim_vector'][1][2])
        print(mosi_data['label'])
        print(mosi_data['output'])
        print(mosi_data['id'])
    if 'mosi_data' in locals():
        del mosi_data
    gc.collect()
    torch.cuda.empty_cache()  # 清空 PyTorch 的 CUDA 缓存

# with open('/data/zhaoxianbing/CMU/datasets/mosi_images.pkl', 'rb') as file:
#     data = pkl.load(file)
# print(data)

#b = torch.nonzero(a<0).squeeze()

#print(a[b])

# # 假设 q 和 k 是两个不同特征矩阵
# q = torch.randn(10, 5,768)  # 10个样本，128维特征
# k = torch.randn(10, 5,128)  # 10个样本，128维特征

# # 计算余弦相似度
# cosine_sim = F.cosine_similarity(q.view(q.shape[0],-1).unsqueeze(1), k.view(k.shape[0],-1).unsqueeze(0), dim=-1)
# print(cosine_sim.shape)
# _, most_similar_1 = torch.topk(cosine_sim, k=2, dim=1,largest=True)
# most_similar_1 = most_similar_1[:, 1]
# print(most_similar_1)
# result = q[most_similar_1]
# print(result.shape)

# a = torch.tensor([1,2,3,4,5,6,7,8,9])
# print(a[torch.tensor([8,7,6,5,4,3,2,1])])
