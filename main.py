import torch
import os
import logging
import argparse
from dotenv import load_dotenv
import utilities
import dataloading
import evaluate_models
from models import load_model_by_name
from metrics import compute_tav, compute_tid


def main(args):
    if args.chtc:
        utilities.chtc_setup([args.dataset], [args.model_name])
    model, model_transforms = load_model_by_name(args.model_name)
    args.model_name = args.model_name.replace("/", "+")
    imagenet_wn_to_name = utilities.imagenet_wordnet_to_name()
    if args.dataset == "ptd":
        dataset = dataloading.PromptedTexturesDataset(
            root="datasets/ptd",
            transform=model_transforms,
            data_classes="all",
            return_paths=True,
        )
    else:
        dataset = dataloading.BetterImageFolder(
            root=f"datasets/{args.dataset}",
            transform=model_transforms,
            split_folder=args.dataset,
            return_paths=True,
        )
    classes = (
        dataset.classes
        if args.dataset == "ptd"
        else [imagenet_wn_to_name[c] for c in dataset.classes]
    )
    dataloader = dataloading.get_dataloader(
        dataset,
        subset=args.subset,
        shuffle_pre_subset=args.shuffle_pre_subset,
        batch_size=args.batch_size,
        data_kwargs={"num_workers": 4} if device == "cuda" else {},
    )
    save_prefix = f"{save_dir}/{args.model_name}_{args.dataset}"
    confidences_df = evaluate_models.evaluate_model(
        model,
        dataloader,
        device,
        save_softmax=args.dataset != "ptd",
        label_classes=classes,
        softmax_save_path=f"{save_prefix}_softmax.csv",
    )
    if args.dataset == "ptd":
        compute_tav(confidences_df).to_csv(f"{save_prefix}_tav.csv")
    else:
        compute_tid(
            confidences_df,
            softmax_path=f"{save_prefix}_softmax.csv",
            tav_path=f"{save_prefix.replace(args.dataset, 'ptd')}_tav.csv",
        ).to_csv(f"{save_prefix}_tid.csv", index=False)
    confidences_df.to_csv(f"{save_prefix}_confidences.csv", index=False)
    if args.chtc:
        confidences_df.to_csv(f"{args.model_name}_{args.dataset}_confidences.csv")


if __name__ == "__main__":
    # Get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to use. Options are 'ptd', 'imagenet-a', or 'imagenet-val'.",
        default=None,
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        help="Model to use. Default weights will be loaded.",
        default="resnet50",
    )
    parser.add_argument(
        "-s",
        "--subset",
        help="Use a subset of the dataset for testing. Either max and min or just min can be specified. By default, the entire dataset is used. If using the PTD, only use this argument for debugging otherwise results will be very misaligned due to sparsity in the TAV computation.",
        nargs=2,
        type=int,
        default=None,
    )
    parser.add_argument(
        "--shuffle_pre_subset",
        action="store_true",
        help="Shuffle the dataset before taking the subset. Default is False. Setting as True will lead to non-determinism and potentially mismatched results.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size to use for testing. Default is 128. If subset is used, the batch size will be adjusted if it exceeds the length of the subset.",
        default=256,
    )
    parser.add_argument(
        "--chtc",
        action="store_true",
        help="Use this flag if running on CHTC at UW-Madison to facilitate data transfer and result saving.",
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    logging.basicConfig(level=logging.INFO)
    base_dir = f"results"
    if args.chtc:
        load_dotenv()
        STAGING_DIR = os.getenv("STAGING_DIR")
        torch.hub.set_dir(".")
        save_dir = f"{STAGING_DIR}/{base_dir}"
        load_dir = save_dir
    else:
        save_dir = base_dir
        load_dir = base_dir
    main(args)
