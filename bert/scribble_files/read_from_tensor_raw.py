import torch
with open('stuff_inside_tensor_test_oracle.txt', 'w') as f:
    # for line in torch.load("../bert_data_cnndm_final/cnndm.test.0.bert.pt"):
    # for line in torch.load("../highlighted_sentence_data_dir/cnndm.train.0.bert.pt"):
    for line in torch.load("../highlighted_sentence_data_dir_20k/oracle_test/cnndm.test.0.bert.pt"):
        f.write(f"{line}\n")