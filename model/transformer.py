import torch
from torch import nn

try:
    from .common import Transformer, LayerNorm
except:
    from common import Transformer, LayerNorm

class TransformerStudent(nn.Module):
    def __init__(self, context_length, vocab_size, transformer_width, transformer_layers, transformer_heads, output_dim):
        super(TransformerStudent, self).__init__()
        
        # context length, vocabulary size, and transformer width
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        
        # initialize the Transformer network
        self.transformer = Transformer(transformer_width, transformer_layers, transformer_heads, self.build_attention_mask())
        
        # add a LayerNorm layer
        self.ln_final = LayerNorm(transformer_width)
        
        # positional embedding
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        
        # text projection
        self.text_projection = nn.Parameter(torch.empty(transformer_width, output_dim))
        
        # initialize parameters
        self.initialize_parameters()

    def initialize_parameters(self):
        # initialize weights of token embedding
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        
        # initialize weights of positional embedding
        nn.init.normal_(self.positional_embedding, std=0.01)
        
        # standard deviation of the project weights
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        
        # standard deviation of the attention weights
        attn_std = self.transformer.width ** -0.5
        
        # standard deviation of the fully connected weights
        fc_std = (2 * self.transformer.width) ** -0.5
        
        # initialize weights of each block
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        
        # initialize weights of text projection if not None
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def encode_text(self, text):
        # apply token embedding
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        
        # add positional embedding
        x = x + self.positional_embedding
        
        # permute dimensions for Transformer input
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # apply Transformer network
        x = self.transformer(x)
        
        # permute dimensions back to original
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        # apply LayerNorm
        x = self.ln_final(x)
        
        # calculate logits with text projection
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def build_attention_mask(self):
        # create empty mask with -inf values
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        
        # set lower diagonal to zero
        mask.triu_(1)
        return mask

    def forward(self, text, image_feature=None):
        """
        Forward pass of the TransformerStudent model.
    
        Args:
            text (torch.Tensor): Tensor of shape [batch_size, seq_length] representing the input text.
            image_feature (torch.Tensor, optional): Tensor of shape [batch_size, feature_size] representing the image feature.
    
        Returns:
            torch.Tensor: If image_feature is None, returns the encoded text feature tensor of shape [batch_size, output_dim].
                          If image_feature is not None, returns the logits tensor of shape [global_batch_size, global_batch_size].
        """
        if image_feature is None:
            return self.encode_text(text)
        text_feature = self.encode_text(text)
    
        logits_per_text = self.calculate_logits(image_feature, text_feature)
        return logits_per_text
    
    def calculate_logits(self, image_feature, text_feature):
        """
        Calculates the logits tensor for the TransformerStudent model.
    
        Args:
            image_feature (torch.Tensor): Tensor of shape [batch_size, feature_size] representing the image feature.
            text_feature (torch.Tensor): Tensor of shape [batch_size, output_dim] representing the encoded text feature.
    
        Returns:
            torch.Tensor: Logits tensor of shape [global_batch_size, global_batch_size].
        """
        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
    
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_feature @ text_feature.t()
    
        # shape = [global_batch_size, global_batch_size]
        return logits_per_image.t()


if __name__ == '__main__':
    text_model = TransformerStudent(77, 49408, 128, 6, 8, 512)
    print(text_model(torch.randint(low=0, high=49409, size=(3, 77))).shape)
