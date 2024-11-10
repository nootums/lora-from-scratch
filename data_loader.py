from transformers import AutoTokenizer
import json

with open("config.json", 'r') as f:
    config = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

def load_json_data(json_loc=None):
    data_loc = json if json_loc else config["data_loc"]
    with open(data_loc, 'r') as f:
        data = json.load(f)
    samples = []
    for i in data:
        samples.append(data[i])
    return samples

def load_dataset_from_list(sample_list:list[str]):
    sep_token = config["sep_token"]
    sep_token_id = tokenizer(sep_token)["input_ids"][-1]
    data = []
    for sentence in sample_list:
        instance = {}
        sentence += tokenizer.eos_token
        tokenized_sentence = tokenizer(sentence)["input_ids"]
        instance["input_tokens"] = tokenized_sentence[:-1]
        instance["output_tokens"] = tokenized_sentence[1:]
        instance["sep_token_index"] = tokenized_sentence.index(sep_token_id)
        data.append(instance)
    return data
    