import pandas as pd
import re
import time
import torch
import torch.utils.data as data
from tqdm.auto import tqdm
from transformers import DistilBertTokenizerFast


TRAIN_TEXT_PATH = "data/training_text"
TRAIN_VARIANTS_PATH = "data/training_variants"
TEST_TEXT_PATH = "data/test_text"
TEST_VARIANTS_PATH = "data/test_variants"


def read_text(path):
    with open(path, "r", encoding="utf8") as f:
        _ = f.readline()
        lines = f.readlines()
        
    texts = [line.split("||")[1] for line in lines]

    return texts


def read_genes_mutations_classes(path):
    df = pd.read_csv(path)

    genes = df["Gene"].to_list()
    mutations = df["Variation"].to_list()
    classes = (df["Class"] - 1).to_list()

    return genes, mutations, classes


def get_estimate_tokens(sentence):
    return len(re.split("\s+", sentence)) * 1.82  # Average number of tokens per word is 1.82


def get_surrounding_text(query, sentences):
    results = []
    tokens = 0

    for i, s in enumerate(sentences):
        if query in s:
            results.append(i)
            tokens += get_estimate_tokens(s)

    results = results[::-1]

    candidate = 0
    while candidate < len(results):
        selected = results[candidate]
        candidate += 1

        after = selected + 1
        if after < len(sentences) and after not in results:
            est_after = get_estimate_tokens(sentences[after])
            if tokens + est_after < 500:
                tokens += est_after
                results.append(after)
        
        # Try the previous sentence
        prev = selected - 1
        if prev >= 0 and prev not in results:
            est_prev = get_estimate_tokens(sentences[prev])
            if tokens + est_prev < 500:
                tokens += est_prev
                results.append(prev)
    
    return sorted(results)


def get_paragraph(text, gene, mutation):
    sentences = re.sub("Fig.", "Fig", text)
    sentences = re.split("\.\s+", sentences)

    search_results = get_surrounding_text(mutation, sentences)

    if len(search_results) == 0:
        search_results = get_surrounding_text(gene, sentences)

    paragraph = ". ".join([sentences[i] for i in search_results])

    return paragraph


class GeneDataset(data.Dataset):
    def __init__(self):
        texts = read_text(TRAIN_TEXT_PATH)
        genes, mutations, classes = read_genes_mutations_classes(TRAIN_VARIANTS_PATH)

        paragraphs = []
        variants = []
        
        print("Extracting sentences...")
        bad_indices = []
        for i, (text, gene, mutation) in enumerate(tqdm(zip(texts, genes, mutations), total=len(texts))):
            p = get_paragraph(text, gene, mutation)

            if p:
                paragraphs.append(p)
            else:
                bad_indices.append(i)

            variants.append(f"{gene} {mutation}")
        
        variants = [v for i, v in enumerate(variants) if i not in bad_indices]
        classes = [c for i, c in enumerate(classes) if i not in bad_indices]

        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

        print("Tokenizing...")
        self.ids = []
        self.labels = []
        for index in tqdm(range(len(classes))):
            text = "[SEP]".join([variants[index], paragraphs[index]])

            inputs = tokenizer.encode_plus(text=text, padding="max_length", 
                return_token_type_ids=True, truncation=True, max_length=512)
            
            self.ids.append(torch.LongTensor(inputs["input_ids"]))
            self.labels.append(torch.LongTensor([classes[index]]))

    def __getitem__(self, index):
        return self.ids[index], self.labels[index]

    def __len__(self):
        return len(self.ids)


def get_train_test_loaders(batch_size, ratio=0.8):
    dataset = GeneDataset()

    train_len = int(len(dataset) * ratio)
    test_len = len(dataset) - train_len

    train_dataset, test_dataset = data.random_split(dataset, [train_len, test_len])

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader


def single_inference(text, gene, mutation):
    paragraph = get_paragraph(text, gene, mutation)

    if not paragraph:
        return None
        
    text = f"{gene} {mutation} [SEP] {paragraph}"

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    ids = tokenizer.encode_plus(text=text, padding="max_length", 
        return_token_type_ids=True, truncation=True, max_length=512)["input_ids"]
    
    return ids


if __name__ == "__main__":
    get_train_test_loaders(4)