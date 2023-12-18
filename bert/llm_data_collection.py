import argparse
import os
import openai
import os
import replicate
import constants
import torch
from nltk.tokenize.treebank import TreebankWordDetokenizer

openai.api_key = constants.OPEN_AI_KEY
os.environ["REPLICATE_API_TOKEN"] = constants.LLAMA



start_offset = 0
end_offset = 2

input_file_path = "./highlighted_sentence_data_dir_20k/oracle_test/cnndm.test.0.bert.pt"
output_dir_path = "./highlighted_sentence_data_dir_20k/oracle_test/gpt_3.5_turbo_instruct_test"
mode = "sentence" #"keywords"
api =  "llama2"#"gpt3"
"""
Used for printing guidance signals using GPT-3.5-turbo-instruct and LLama2-Chat
"""

if __name__ == "__main__":
    if mode == "sentence":
        for i, line in enumerate(torch.load(input_file_path)):
            if i >= start_offset and i < end_offset:
                print("Processing line", i)
                src_text = line["src_txt"]
                print("Source text")
                print(src_text)
                print()
                print("Cleaned source text")
                cleaned_src_text = " ".join([TreebankWordDetokenizer().detokenize(sentence.split(" ")) for sentence in src_text])
                print(cleaned_src_text)
                print()
                if api == "gpt3":
                    print("Calling GPT-3.5-turbo-instruct")
                    sentence_prompt = f"""
                    Your job is to be a helpful guide system in
                    document summarization. 
                    Here is an article:
                    SOURCE: {cleaned_src_text}
                    
                    Return the most important sentences from
                    SOURCE that are important to generating a
                    summary. Return them as a period separated list of
                    sentences.
                    """
                    response = openai.Completion.create(
                    engine="gpt-3.5-turbo-instruct",
                    prompt=sentence_prompt,
                    max_tokens=512,
                    n=1,
                    stop=None,
                    temperature=0.5,
                    )
                    print(response.choices[0].text.strip())
                    print()
                elif api == "llama2":
                    print("Calling LLAMA2")
                    system_prompt = "Your job is to be a helpful guide system in document summarization."
                    sentence_prompt = f"""
                    Here is an article:
                    SOURCE: {cleaned_src_text}
                    
                    Return the most important sentences from
                    SOURCE that are important to generating a
                    summary. Return them as a period separated list of
                    sentences.
                    """
                    output = replicate.run(
                            "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
                            input = {
                                "debug": False,
                                "top_k": 50,
                                "top_p": 1,
                                "prompt": sentence_prompt,
                                "temperature": 0.5,
                                "system_prompt": system_prompt,
                                "max_new_tokens": 512,
                                "min_new_tokens": -1
                            }
                    )
                    for item in output:
                        print(item, end="")
                    print()
                    
            else:
                break
    
    elif mode == "keywords":
        for i, line in enumerate(torch.load(input_file_path)):
            print("Processing line", i)
            src_text = line["src_txt"]
            print("Source text")
            print(src_text)
            print()
            print("Cleaned source text")
            cleaned_src_text = " ".join([TreebankWordDetokenizer().detokenize(sentence.split(" ")) for sentence in src_text])
            print(cleaned_src_text)
            print()
            if api == "gpt3":
                print("Calling GPT-3.5-turbo-instruct")
                keywords_prompt = f"""
                    Your job is to be a helpful guide system in document summarization. 
                    Here is an article:
                    SOURCE: {cleaned_src_text}
                    Return the most important keywords from SOURCE that are important for generating a summary. 
                    Return them as a comma seperated list of keywords.
                    """
                response = openai.Completion.create(
                engine="gpt-3.5-turbo-instruct",
                prompt=sentence_prompt,
                max_tokens=512,
                n=1,
                stop=None,
                temperature=0.5,
                )
                print(response.choices[0].text.strip())
                print()
            elif api == "llama2":
                print("Calling LLAMA2")
                system_prompt = "Your job is to be a helpful guide system in document summarization."
                keywords_prompt = f"""
                    Here is an article:
                    SOURCE: {cleaned_src_text}
                    Return the most important keywords from SOURCE that are important for generating a summary. 
                    Return them as a comma seperated list of keywords.
                    """
                output = replicate.run(
                        "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
                        input = {
                            "debug": False,
                            "top_k": 50,
                            "top_p": 1,
                            "prompt": keywords_prompt,
                            "temperature": 0.5,
                            "system_prompt": system_prompt,
                            "max_new_tokens": 512,
                            "min_new_tokens": -1
                        }
                )
                for item in output:
                    print(item, end="")
                print()
    