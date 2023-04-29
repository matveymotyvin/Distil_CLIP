# Import necessary libraries
import json
from pathlib import Path
from clip import tokenize
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class TextDataset(Dataset):
    # Define the constructor for the TextDataset class
    def __init__(self, cache_dir, data_dir, train=True, overwrite=False,
                 img_mean=(0.485, 0.456, 0.406),
                 img_std=(0.229, 0.224, 0.225)):
        # Call the constructor of the base class
        super(TextDataset, self).__init__()

        # Store the given arguments as instance variables
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.train = train
        self.tokenizer = tokenize

        # If the object is being initialized for training
        if self.train:
            # Load the tokenized text and save it to instance variable
            self.tokenize_text = self.load(overwrite)

        # If the object is being initialized for validation
        if not self.train:
            # Load image mean and standard deviation and save them to instance variables
            self.img_mean = img_mean
            self.img_std = img_std

            # Load captions, sentences, and path list and save them to instance variables
            self.sentences, self.captions, self.path_list = self.load(overwrite)

    # Define the process method to tokenize the raw text
    def process(self):
        # Create an empty list to store raw text
        raw_text = []

        # If the object is being initialized for training
        if self.train:
            # Load the train data and concatenate the captions and raw text
            coco_file = self.data_dir / 'COCO' / 'annotations' / 'captions_train.json'
            cc_file = self.data_dir / 'CC' / 'Train_GCC-training.tsv'

            with cc_file.open('r', encoding='utf8') as f:
                for content in f.readlines():
                    raw_text.append(content.split('\t')[0])

            with coco_file.open('r', encoding='utf8') as f:
                res = json.load(f)
                for annotation in res['annotations']:
                    raw_text.append(annotation['caption'])

            # Print the number of raw text data and start tokenizing
            print('All data: {} Begin tokenizing...'.format(len(raw_text)))
            tokenize_text = []

            # Iterate over raw text and tokenize them
            for text in tqdm(raw_text):
                tokenize_text.append(self.tokenizer(text, truncate=True).squeeze())

            # Stack the tokenized text and return it
            return torch.stack(tokenize_text)

        # If the object is being initialized for validation
        else:
            # Load the validation image file list path and create empty lists to store captions, sentences, and path list
            val_image_file_list_path = self.data_dir / 'COCO' / 'validation'
            path_list = []
            captions = []
            sentences = []
            file_dir = self.data_dir / 'COCO' / 'annotations' / 'captions_validation.json'

            # Load captions, sentences, and path list from the validation file
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
def load(self, overwirite):
    """Load dataset from cache directory, or process and save it in cache directory.

    Args:
        overwirite (bool): Whether to overwrite cache file if it already exists.

    Returns:
        data (list or tuple): Processed data from dataset.

    """
    # Define cache path based on whether dataset is for training or validation.
    cache_path = self.cache_dir / 'cache-train.pth' if self.train else self.cache_dir / 'cache-val.pth'

    # Create cache directory if it does not exist.
    if not self.cache_dir.exists():
        self.cache_dir.mkdir()

    if overwirite or not cache_path.exists():
        # If dataset is for training, tokenize text and save in cache file.
        # If dataset is for validation, process sentences and captions, and save in cache file.
        print('Rewrite/There is no cache file, start processing the file')
        if self.train:
            tokenize_text = self.process()
            torch.save({'data_set': tokenize_text}, cache_path)
            return tokenize_text
        else:
            sentences, captions, path_list = self.process()
            torch.save({
                'data_set': [
                    sentences,
                    captions,
                    path_list
                ]
            }, cache_path)
            return sentences, captions, path_list
    else:
        # Load data from cache file.
        print('Load cache files directly')
        data = torch.load(cache_path)['data_set']
        print(' Loading is complete')
        return data


    def __len__(self):
        """Get the length of dataset.
    
        Returns:
            length (int): Length of dataset.
    
        """
        if self.train:
            return len(self.tokenize_text)
        else:
            return len(self.path_list)
    
    
    def __getitem__(self, idx):
        """Get a sample from the dataset.
    
        Args:
            idx (int): Index of the sample to retrieve.
    
        Returns:
            sample (tuple): A tuple containing image tensor, caption, and sentence.
    
        """
        if self.train:
            return self.tokenize_text[idx]
    
        # If dataset is for validation, open image at the given index and preprocess it.
        path = self.path_list[idx]
        img = Image.open(path).convert('RGB')
        trans = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(self.img_mean, self.img_std)
        ])
        img_tensor = trans(img)
        return img_tensor, self.captions[idx], self.sentences[idx]


if __name__ == '__main__':
    from clip import tokenize
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt

    # Load dataset for training and print first batch.
    text_data = TextDataset('/home/pyz/data/cache', '/home/pyz/data', True, tokenize, True)
    for data in DataLoader(text_data, batch_size=2):
        print(data)
        break
    text_data = TextDataset('/home/pyz/data/cache', '/home/pyz/data', False, tokenize, False)
    for data in DataLoader(text_data, batch_size=1, shuffle=True):
        image, caption, sentence = data
        plt.imshow(image.squeeze(0).permute(1, 2, 0))
        print(caption, sentence)
        break


