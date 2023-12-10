import torch
with open('stuff_inside_tensor.txt', 'w') as f:
    for line in torch.load("./bert_data_cnndm_final/cnndm.train.0.bert.pt"):
        f.write(f"{line}\n")