import pandas as pd
import time
import torch
import torch.utils.data as data
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


def read_variants(path):
    df = pd.read_csv(path)

    classes = df["Class"].to_list()
    classes = [n - 1 for n in classes]

    return classes


class GeneDataset(data.Dataset):
    def __init__(self):
        self.texts = read_text(TRAIN_TEXT_PATH)
        self.variants = read_variants(TRAIN_VARIANTS_PATH)
        
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def __getitem__(self, index):
        inputs = self.tokenizer.encode_plus(text=self.texts[index], padding="max_length", 
            return_token_type_ids=True, truncation=True, max_length=512)
        
        ids = torch.LongTensor(inputs["input_ids"])
        mask = torch.LongTensor(inputs["attention_mask"])

        variants = torch.LongTensor([self.variants[index]])

        return ids, mask, variants

    def __len__(self):
        return len(self.texts)


def get_train_test_loaders(batch_size, ratio=0.8):
    dataset = GeneDataset()

    train_len = int(len(dataset) * ratio)
    test_len = len(dataset) - train_len

    train_dataset, test_dataset = data.random_split(dataset, [train_len, test_len])

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader
