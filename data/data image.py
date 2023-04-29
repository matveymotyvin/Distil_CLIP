import json
import os
import os.path as op

from PIL import Image
from clip import tokenize
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(
        self,
        data_dir: str = 'data/ref',
        train: bool = True,
        no_augment: bool = True,
        aug_prob: float = 0.5,
        img_mean: tuple = (0.485, 0.456, 0.406),
        img_std: tuple = (0.229, 0.224, 0.225)
    ):
        """
        A PyTorch Dataset for images.

        Args:
            data_dir (str): The path to the directory containing the data.
            train (bool): Whether to use the training data or validation data.
            no_augment (bool): Whether to perform data augmentation on the training data.
            aug_prob (float): The probability of applying data augmentation on a given image.
            img_mean (tuple): The mean values of the RGB channels for normalization.
            img_std (tuple): The standard deviation values of the RGB channels for normalization.
        """
        super(ImageDataset, self).__init__()

        # Update the object's attributes with the arguments passed to the constructor.
        self.__dict__.update(locals())

        # Determine whether to apply data augmentation.
        self.augmentation = train and not no_augment

        # Initialize path_list to None.
        self.path_list = None

        # If not using the training data, initialize the tokenizer.
        if not train:
            self.tokenizer = tokenize

        # Process the data.
        self.process_data()

    def process_data(self):
        """
        Process the data based on the mode (train or validation) specified in the constructor.
        """
        if self.train:
            # If using the training data, get the paths to all images in the images subdirectory.
            train_image_path = op.join(self.data_dir, 'images')
            self.path_list = [op.join(train_image_path, i) for i in os.listdir(train_image_path)]
        else:
            # If using the validation data, get the paths to all images in the validation subdirectory.
            val_image_path = op.join(self.data_dir, 'COCO', 'validation')
            self.path_list = []
            self.tokenized_captions = []
            self.sentence = []
            self.annotations_dir = op.join(self.data_dir, 'COCO', 'annotations')
            with open(op.join(self.annotations_dir, 'captions_validation.json'), 'r') as f:
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
                    self.sentence.append(caption)
                    self.tokenized_captions.append(self.tokenizer(caption).squeeze())
                    self.path_list.append(op.join(val_image_path, file_name))

    def __len__(self):
        """
        Get the number of images in the dataset.
        """
        return len(self.path_list)

    def __getitem__(self, idx):
        """
        Get the image, tokenized caption, and original sentence at the specified index.

        Args:
            idx (int): The index of the item to get.

        Returns:img_tensor
	  """
        path = self.path_list[idx]

        img = Image.open(path).convert('RGB')
        trans = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(self.aug_prob),
            transforms.RandomVerticalFlip(self.aug_prob),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(self.img_mean, self.img_std),
        ]) if self.train else transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(self.img_mean, self.img_std)
        ])

        img_tensor = trans(img)

        if self.train:
            return img_tensor
        else:
            return img_tensor, self.tokenized_captions[idx], self.sentence[idx]
