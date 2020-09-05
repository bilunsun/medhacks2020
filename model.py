import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import DistilBertModel


class GeneClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = nn.Linear(768, 768)
        self.classifier = nn.Linear(768, 9)

        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, mask=None):
        hidden = self.bert(input_ids, mask)[0]

        cls_tensors = hidden[:, 0]

        x = self.pre_classifier(cls_tensors)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.classifier(x)

        return x


if __name__ == "__main__":
    model = GeneClassifier()
    print(model)