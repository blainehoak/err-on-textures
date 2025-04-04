import numpy as np
import pandas as pd
import utilities
from sklearn.metrics.pairwise import cosine_similarity


def compute_tav(confidences_df):
    conf_matrix = (
        confidences_df[["labels", "preds", "confidences"]]
        .groupby(["labels", "preds"])
        .count()
        .unstack()
        .fillna(0)
    )
    conf_matrix.columns = conf_matrix.columns.droplevel()
    # add the missing columns to conf_matrix
    df = conf_matrix.loc[:, (conf_matrix != 0).any(axis=0)]
    k, c = df.shape
    # Convert DataFrame to numpy array for easier manipulation
    N = df.to_numpy()

    # Normalize N to get probability distribution P
    PTij = N / np.sum(N, axis=1, keepdims=True)

    # weight the entropy by the number of classifications for each object class
    POij = N / np.sum(N, axis=0)

    # Calculate entropy for each texture class
    THi = -np.nansum(PTij * np.emath.logn(n=c, x=PTij + 1e-9), axis=1)

    OHj = -np.nansum(POij * np.emath.logn(n=k, x=POij + 1e-9), axis=0)

    textureness = POij * (1 - THi)[:, np.newaxis] * PTij * (1 - OHj)
    df_tav = pd.DataFrame(textureness, index=df.index, columns=df.columns)
    imagenet_classes = utilities.imagenet_classes_list()
    # if there are any class names that are not in the columns, add them with all zeros
    missing_cols = list(set(imagenet_classes).difference(set(df_tav.columns)))
    # create a dataframe that has the missing columns and has 0 for all values
    missing_df = pd.DataFrame(
        np.zeros((df_tav.shape[0], len(missing_cols))),
        columns=missing_cols,
        index=df_tav.index,
    )
    return pd.concat([df_tav, missing_df], axis=1)


def compute_tid(confidences, softmax_path, tav_path):
    softmaxes = pd.read_csv(softmax_path)
    tav_matrix = pd.read_csv(tav_path, index_col=0)

    # order the columns to be the same as df
    tav_matrix = tav_matrix[softmaxes.columns]
    text_conf_np = tav_matrix.to_numpy()
    text_conf_np = text_conf_np / text_conf_np.sum(axis=1)[:, np.newaxis]
    # get the cosine similarity between every row in df and text_df
    cos_sim = cosine_similarity(softmaxes.to_numpy(), text_conf_np)
    # turn the cosine similarity into a dataframe
    cos_sim_df = pd.DataFrame(cos_sim, columns=tav_matrix.index)
    cos_sim_pred_df = pd.DataFrame(
        {
            "cos_sim_max": cos_sim_df.idxmax(axis=1),
            "cos_sim_val": cos_sim_df.max(axis=1),
        }
    )
    # get the texture class that has the highest cosine similarity for each object class and the value of the cosine similarity. merge this with the df_og dataframe
    cos_sim_pred_df = cos_sim_pred_df.merge(
        confidences[["img_path"]],
        left_index=True,
        right_index=True,
    )
    return cos_sim_pred_df
