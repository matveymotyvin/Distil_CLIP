import torch
from torch import nn

# Import local modules
try:
    from .model_teacher import TeacherModel
    from .transformer import TransformerStudent
except ImportError:
    from model_teacher import TeacherModel
    from transformer import TransformerStudent


class ModelTextDistilled(nn.Module):
    def __init__(self, teacher_name, context_length, vocab_size, transformer_width, transformer_layers,
                 transformer_heads, output_dim):
        super(ModelTextDistilled, self).__init__()

        # Instantiate teacher model
        self.teacher = TeacherModel(teacher_name=teacher_name)
        # Freeze teacher model parameters
        for param in self.parameters():
            param.requires_grad = False

        # Instantiate student model
        self.student = TransformerStudent(context_length, vocab_size, transformer_width, transformer_layers,
                                          transformer_heads, output_dim)

    def forward(self, text):
        # Pass input text through student model
        stu_encode = self.student(text)

        # Pass input text through teacher model with no gradient tracking
        with torch.no_grad():
            tea_encode = self.teacher.encode_text(text).float()

        # Return student and teacher encodings
        return stu_encode, tea_encode


if __name__ == '__main__':
    # Instantiate model and print output shape
    text_model = ModelTextDistilled('ViT-B/32', 77, 49408, 128, 8, 8, 512).to('cuda:0')
    print(text_model(torch.randint(low=0, high=49409, size=(3, 77)).cuda())[0].shape)
