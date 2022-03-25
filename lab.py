#!/usr/bin/env python
# -*- coding = utf-8 -*-
# @Time : 2022/3/25 18:45
# @Author : ZiJian Liu @ SUSE
# @File : lab.py
# @Software: PyCharm

import torch

save_ids = []
save_scores = []

print(save_ids)
# print(save_ids.shape)

output_ids = torch.tensor([[1, 2]])
print(output_ids.shape)

save_ids.extend(output_ids)
print(save_ids)
# print(save_ids.shape)

output_ids2 = torch.tensor([[1, 2, 3], [2, 3, 4]])
print(output_ids2.shape)

save_ids.extend(output_ids2)
print(save_ids)
# print(save_ids.shape)

print(save_ids[1])
print(save_ids[1].shape)


# flat = torch.tensor([True, True, False])
# print(flat.all())

# output_score = torch.tensor([1, 2])
# print(output_score.shape)
# save_scores.extend(output_score)
# print(save_scores)
# print(torch.tensor(save_scores).argmax().item())