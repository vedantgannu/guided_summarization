import torch
from others.tokenization import BertTokenizer
import nltk



src_data_path = "./highlighted_sentence_data_dir_20k/oracle_test/cnndm.test.0.bert.pt"

start_offset = 0
end_offset = 2


if __name__ == "__main__":
    data_path = "./highlighted_sentence_data_dir_20k/oracle_test/cnndm.test.0.bert.pt"
    for i, line in enumerate(torch.load(data_path)):
        if i >= start_offset and i < end_offset:
            print("Prcocessing line", i)
            print()
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            tokens = tokenizer.convert_ids_to_tokens(line["z"])
            tokens = list(filter(lambda x: x != "[CLS]" or x != "[SEP]", tokens))
            print(" ".join(tokens))
            print()
            print("Source text")
            print("".join(line["src_txt"]))
            print()
            print("Target text")
            print(line["tgt_txt"])
            print()
        else:
            break