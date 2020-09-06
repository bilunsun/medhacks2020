from data_utils import get_single_inference_ids


def get_single_inference_class(text, gene, mutation, model):
    model.eval()

    ids = get_single_inference_ids(text, gene, mutation)
    prediction = model(ids)

    return prediction.to_list()
