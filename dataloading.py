import torch
import numpy as np
import os
import pathlib
from typing import Any, Callable, Optional, Tuple
import PIL.Image
from torchvision.datasets import VisionDataset, ImageFolder


class PromptedTexturesDataset(VisionDataset):
    """Prompted Textures Dataset

    Args:
        root (string): Root directory of the dataset.
        prompt_file (string): Path to the json file with translations from hashes to prompts and nsfw counts.
        transform (callable, optional): A function/transform that  takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        data_classes="all",
        return_paths=False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._base_folder = pathlib.Path(self.root)
        self._data_folder = self._base_folder

        self._images_folder = self._base_folder / "images"
        self.return_paths = return_paths

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )
        self._image_files = []
        curr_classes = []
        with open(self._base_folder / f"metafile.txt") as f:
            for line in f:
                if data_classes == "all" or line.strip().split("/")[0] == data_classes:
                    self._image_files.append(self._images_folder / line.strip())
                    curr_classes.append(line.strip().split("/")[0])
        self.classes = list(set(curr_classes))
        self.all_classes = os.listdir(self._images_folder)
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        self._labels = [
            self.class_to_idx[img_file.parts[-2]] for img_file in self._image_files
        ]

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)
        return_set = (image, label)
        # convert to string if return_paths is True
        return_set = (
            return_set + (str(image_file).split("images/")[-1],)
            if self.return_paths
            else return_set
        )
        return return_set

    def _check_exists(self) -> bool:
        return os.path.exists(self._data_folder) and os.path.isdir(self._data_folder)


class BetterImageFolder(ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        return_paths=False,
        split_folder="imagenet-val/",
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.return_paths = return_paths
        self.split_folder = (
            split_folder + "/" if split_folder[-1] != "/" else split_folder
        )

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file, label = self.samples[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)
        if self.return_paths:
            return image, label, str(image_file).split(self.split_folder)[-1]
        return image, label


def get_dataloader(
    dataset,
    subset=None,
    batch_size=128,
    shuffle_pre_subset=False,
    dataloader_shuffle=False,
    disable_batch_size_adjustment=False,
    data_kwargs={},
):
    # data_kwargs = {}
    if subset is not None:
        if len(subset) == 1:
            subset = (subset[0], len(dataset))
        if shuffle_pre_subset:
            subset_idx = np.random.permutation(len(dataset))[subset[0] : subset[1]]
        else:
            subset_idx = list(range(subset[0], subset[1]))
        dataset = torch.utils.data.Subset(dataset, subset_idx)
        batch_size = min(batch_size, len(dataset))
    if len(dataset) % batch_size != 0 and not disable_batch_size_adjustment:
        # calculate a batch size that will evenly divide the data
        batch_size = len(dataset) // (len(dataset) // batch_size)
        print("Batch size adjusted to", batch_size)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=dataloader_shuffle,
        **data_kwargs,
    )
    return dataloader


def load_single_sample_from_path(
    img_path: str, transform: Optional[Callable] = None
) -> Tuple[Any, Any]:
    image = load_image(img_path, resize=False)
    return load_single_sample(image, transform), img_path.split("/")[-2]


def load_single_sample(
    image: PIL.Image.Image, transform: Optional[Callable] = None
) -> Any:
    if transform:
        image = transform(image).unsqueeze(0)
    return image


def load_image(image_path: str, resize: bool = True) -> PIL.Image.Image:
    image = PIL.Image.open(image_path).convert("RGB")
    if resize:
        return image.resize((256, 256))
    return image
