import utilities
import torch
import pandas as pd
import numpy as np


def evaluate_model(
    model,
    dataloader,
    device,
    save_softmax=False,
    label_classes=None,
    softmax_save_path=None,
):
    model = model.eval().to(device)
    confidences = {"confidences": [], "labels": [], "preds": []}
    img_paths = []
    if save_softmax and softmax_save_path is None:
        raise ValueError("softmax_save_path must be specified if save_softmax is True")
    if save_softmax:
        softmax_file = open(softmax_save_path, "w")
        softmax_file.write(",".join(utilities.imagenet_classes_list()) + "\n")
    for i, batch in enumerate(dataloader):
        utilities.progressBar(i, len(dataloader), "Batch: ")
        x, y = batch[0].to(device), batch[1].to(device)
        if len(batch) > 2:
            img_paths.extend(batch[-1])
        softmax = model_forward_pass(model, x)
        if save_softmax:
            np.savetxt(softmax_file, softmax.cpu().numpy(), delimiter=",")
        conf, y_pred = torch.max(softmax, 1)
        confidences["confidences"].append(conf.cpu().numpy())
        confidences["labels"].append(y.cpu().numpy())
        confidences["preds"].append(y_pred.cpu().numpy())
    if save_softmax:
        softmax_file.close()
    confidences = {k: np.concatenate(v) for k, v in confidences.items()}
    if len(img_paths) > 0:
        print(len(img_paths))
        confidences["img_path"] = img_paths
    confidences_df = pd.DataFrame(confidences)
    imagenet_classes = utilities.imagenet_classes_list()
    pred_num_to_name = {i: imagenet_classes[i] for i in range(len(imagenet_classes))}
    labels_num_to_name = (
        {i: c for i, c in enumerate(dataloader.dataset.classes)}
        if label_classes is None
        else {i: c for i, c in enumerate(label_classes)}
    )
    confidences_df["preds"] = confidences_df["preds"].replace(pred_num_to_name)
    confidences_df["labels"] = confidences_df["labels"].replace(labels_num_to_name)
    print(
        f"Model accuracy: {np.mean(confidences_df['preds'] == confidences_df['labels'])}"
    )
    return confidences_df


def evaluate_model_basic_stats(model, dataloader, device):
    model = model.eval().to(device)
    acc, avg_conf = 0, 0
    for i, batch in enumerate(dataloader):
        x, y = batch[0].to(device), batch[1].to(device)
        softmax = model_forward_pass(model, x)
        conf, y_pred = torch.max(softmax, 1)
        acc += (y_pred == y).sum().item()
        avg_conf += conf.sum().item()
    acc /= len(dataloader.dataset)
    avg_conf /= len(dataloader.dataset)
    confidences_df = pd.DataFrame({"accuracy": [acc], "avg_confidence": [avg_conf]})
    return confidences_df


def model_forward_pass(model, x, return_softmax=True):
    with torch.no_grad():
        softmax = model(x)
        if not isinstance(softmax, torch.Tensor):
            # FOR HUGGINGFACE SUPPORT - STILL IN PROGRESS
            softmax = softmax.logits
    if return_softmax:
        softmax = softmax.softmax(dim=1)
    return softmax


def eval_batch_softmax(model, x, device):
    model = model.eval().to(device)
    softmax = model_forward_pass(model, x.to(device), return_softmax=True)
    return softmax
