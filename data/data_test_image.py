import json
import os
from pathlib import Path

from PIL import Image
from clip import tokenize
from torch.utils.data import Dataset


class TestImageDataset(Dataset):
    """Dataset class for image testing."""
    
    def __init__(self, file_path: str, preprocess):
        """
        Args:
            file_path (str): path to directory containing images
            preprocess (Callable): function for preprocessing images
        """
        super(TestImageDataset, self).__init__()
        self.file_path = file_path
        self.file_list = os.listdir(file_path)
        self.preprocess = preprocess

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.file_list)

    def __getitem__(self, item) -> tuple:
        """
        Get an item of the dataset.

        Args:
            item (int): index of the item to retrieve

        Returns:
            tuple: a tuple containing the filename and the preprocessed image
        """
        photo_path = os.path.join(self.file_path, self.file_list[item])
        with Image.open(photo_path).convert('RGB') as image:
            return self.file_list[item], self.preprocess(image)


class TestDataset(Dataset):
    """Dataset class for testing."""
    
    def __init__(self, data_dir: str, preprocess):
        """
        Args:
            data_dir (str): path to directory containing data
            preprocess (Callable): function for preprocessing images
        """
        super(TestDataset, self).__init__()
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenize
        self.img_mean = (0.485, 0.456, 0.406)
        self.img_std = (0.229, 0.224, 0.225)
        self.sentences, self.captions, self.path_list = self.process()
        self.preprocess = preprocess

    def process(self) -> tuple:
        """Process the dataset and return the captions, sentence embeddings, and image paths as lists."""
        val_image_file_list_path = self.data_dir / 'COCO' / 'validation'
        path_list = []
        captions = []
        sentences = []
        file_dir = self.data_dir / 'COCO' / 'annotations' / 'captions_validation.json'
        with file_dir.open('r', encoding='utf8') as f:
            data = json.load(f)
        images = data['images']
        id2caption = {}
        id2filename = {}
        for image in images:
            id2filename[image['id']] = image['file_name']
        for annotation in data['annotations']:
            id2caption[annotation['image_id']] = annotation['caption']
        for id, file_name in id2filename.items():
            caption = id2caption.get(id, None)
            if caption:
                sentences.append(caption)
                captions.append(self.tokenizer(caption).squeeze())
                path_list.append(val_image_file_list_path / file_name)

        return sentences, captions, path_list

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.path_list)

    def __getitem__(self, idx) -> tuple:
        """
        Get an item of the dataset.

        Args:
            idx (int): index of the item to retrieve

        Returns:
            tuple: a tuple containing the filename, preprocessed image, caption, and sentence
        """
        path = self.path_list[idx]
        with Image.open(path).convert('RGB') as img:
            img_tensor = self.preprocess(img)
            return self.path_list[idx].name, img_tensor, self.captions[idx], self.sentences[idx