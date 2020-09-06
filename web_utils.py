from data_utils import get_single_inference_ids
import torch


def get_single_inference_class(text, gene, mutation, model):
    model.eval()

    ids = get_single_inference_ids(text, gene, mutation)
    ids = torch.LongTensor(ids).view(1, -1)
    prediction = model(ids).flatten()

    return prediction.tolist()
