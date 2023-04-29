import clip
import torch
from torch import nn


class TeacherModel(nn.Module):
    def __init__(self, teacher_name, device='cuda'):
        super().__init__()
        # Load the CLIP model and its associated preprocess function
        self.model_teacher, self.preprocess = clip.load(teacher_name, device=device)

    def encode_image(self, image):
        # Get image features using CLIP's encode_image method
        return self.model_teacher.encode_image(image)

    def encode_text(self, text):
        # Get text features using CLIP's encode_text method
        return self.model_teacher.encode_text(text)

    def forward(self, image, text):
        # Get the encoded image and text features
        re_image_features = self.encode_image(image)
        re_text_features = self.encode_text(text)

        # Normalize the image and text features
        image_features = re_image_features / re_image_features.norm(dim=-1, keepdim=True)
        text_features = re_text_features / re_text_features.norm(dim=-1, keepdim=True)

        # Calculate logits per image
        logit_scale = self.model_teacher.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        # Return encoded image and text features, and logits per image
        return re_image_features, re_text_features, logits_per_image


if __name__ == '__main__':
    # Create an instance of the TeacherModel with the ViT-B/32 model
    teacher = TeacherModel("ViT-B/32")
