import torch
from torch import nn

from my_package.model_teacher import TeacherModel
from my_package.model_student import VitStudent


class ModelImageDistilled(nn.Module):
    def __init__(self, teacher_name, input_resolution, patch_size, width, layers, heads, output_dim):
        """Initialize the ModelImageDistilled.

        Args:
            teacher_name (str): The name of the teacher model.
            input_resolution (int): The resolution of the input image.
            patch_size (int): The size of the patches.
            width (int): The number of channels in the intermediate and output layers.
            layers (int): The number of transformer layers.
            heads (int): The number of attention heads.
            output_dim (int): The output dimensionality of the model.
        """
        super(ModelImageDistilled, self).__init__()
        self.teacher = TeacherModel(teacher_name=teacher_name)
        for param in self.parameters():
            param.requires_grad = False

        self.student = VitStudent(input_resolution, patch_size, width, layers, heads, output_dim)

    def forward(self, image):
        """Defines the forward pass of the model.

        Args:
            image (torch.Tensor): The input image.

        Returns:
            A tuple containing the student and teacher encoded images.
        """
        stu_encode = self.student(image)
        with torch.no_grad():
            tea_encode = self.teacher.encode_image(image).float()
        return stu_encode, tea_encode
