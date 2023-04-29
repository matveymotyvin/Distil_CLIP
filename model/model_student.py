import torch
from torch import nn
from common import VisionTransformer as CommonVisionTransformer

class VitStudent(nn.Module):
    def __init__(self, input_resolution, patch_size, width, layers, heads, output_dim):
        """
        Initialize a VitStudent object.

        Args:
            input_resolution (int): Input resolution (image size).
            patch_size (int): Patch size to split the image into.
            width (int): Width of each transformer layer.
            layers (int): Number of transformer layers.
            heads (int): Number of transformer heads.
            output_dim (int): Dimensionality of the output embedding.
        """
        super(VitStudent, self).__init__()

        # Instantiate a CommonVisionTransformer object
        self.student_model = CommonVisionTransformer(input_resolution=input_resolution,
                                                     patch_size=patch_size,
                                                     width=width,
                                                     layers=layers,
                                                     heads=heads,
                                                     output_dim=output_dim)
        
        # Create a learnable parameter to scale logits
        self.logit_scale = nn.Parameter(torch.ones([]))

    def encode_image(self, image):
        """
        Encode an image into a feature representation.

        Args:
            image (tensor): Input image tensor.

        Returns:
            tensor: Encoded image tensor.
        """
        return self.student_model(image)

    def calculate_logits(self, image_feature, text_feature):
        """
        Calculate logits given image and text features.

        Args:
            image_feature (tensor): Feature representation of an image.
            text_feature (tensor): Feature representation of text.

        Returns:
            tensor: Calculated logits.
        """
        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)

        logits_per_image = image_feature @ text_feature.t()
        return logits_per_image

    def forward(self, image, text_feature=None):
        """
        Forward pass of the VitStudent model.

        Args:
            image (tensor): Input image tensor.
            text_feature (tensor, optional): Feature representation of text. Defaults to None.

        Returns:
            tensor: Logits if text_feature is not provided, else feature representation of the image.
        """
        if text_feature is None:
            return self.student_model(image)
        else:
            return self.calculate_logits(self.student_model(image), text_feature) * self.logit_scale.exp()
