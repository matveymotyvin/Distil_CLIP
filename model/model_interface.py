import importlib
import inspect
import pytorch_lightning as pl
import torch
import torch.optim.lr_scheduler as lrs
from torch import nn
from torch.nn import functional as F
from torchmetrics.functional import accuracy
from .utils import cal_logits


class MInterface(pl.LightningModule):
    def __init__(self, model_name, loss, lr, t, weight, loss_scale, **kargs):
        super().__init__()
        self.loss_function = {}  # A dictionary to hold different loss functions
        self.save_hyperparameters()  # Save all the hyperparameters for future reference
        self.load_model()  # Load the PyTorch model from a module
        self.configure_loss()  # Configure the loss functions for the model

    def forward(self, img):
        return self.model(img)

    def on_train_epoch_start(self):
        # Print the current epoch number
        self.print('current epoch: {}'.format(self.current_epoch))
        if self.current_epoch > 100:
            self.hparams.t = 2  # If the current epoch number is greater than 100, set the value of t to 2

    def training_step(self, batch, batch_idx):
        stu_encode, tea_encode = self(batch)  # Get the student and teacher encodings from the batch data
        return self.cal_loss(stu_encode, tea_encode, 'train')  # Calculate the loss for the training step

    def validation_step(self, batch, batch_idx):
        img_tensor, captions, sentence = batch

        # Calculate the student and teacher logits and encodings for the batch data
        (stu_logits_text, tea_logits_text, stu_logits_img, tea_logits_img), stu_encode, tea_encode = cal_logits(
            self.hparams.model_name, self.model, captions, img_tensor)

        self.cal_loss(stu_encode, tea_encode, 'val')  # Calculate the loss for the validation step

        if self.hparams.model_name == 'model_distil_text':
            stu_logits, tea_logits = stu_logits_text, tea_logits_text
        else:
            stu_logits, tea_logits = stu_logits_img, tea_logits_img

        # Apply softmax to the student and teacher logits along the last dimension
        stu_logits, tea_logits = stu_logits.softmax(dim=-1), tea_logits.softmax(dim=-1)

        label = torch.arange(stu_encode.shape[0], device=self.device)  # Create a label tensor
        k_list = [i for i in [1, 2, 3, 4, 5, 10, 20, 30, 50] if i < stu_encode.shape[0]]  # Get a list of k values to compute accuracy for
        for k in k_list:
            if k == 1:
                # Compute top-1 accuracy
                acc = accuracy(stu_logits, label, top_k=1, task='multiclass', num_classes=stu_encode.shape[0])
                self.log('val_acc', acc, on_epoch=True, on_step=False, prog_bar=False, sync_dist=True)
            else:
                # Compute top-k accuracy for k > 1
                acc = accuracy(stu_logits, label, top_k=k, task='multiclass', num_classes=stu_encode.shape[0])
                self.log('val_acc/top{}'.format(k), acc, on_epoch=True, on_step=False, prog_bar=False, sync_dist=True)

        return

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        self.print('')  

    # Define a function to configure optimizers
    def configure_optimizers(self):
        # Check if weight decay is present in hyperparameters, if yes, assign it to variable weight_decay else 0
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        
        # Define an AdamW optimizer with the given learning rate, weight decay and model parameters
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)
    
        # If no learning rate scheduler is specified, return the optimizer
        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            # If a learning rate scheduler is specified, create it based on the given scheduler type
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                # Raise an error if an invalid scheduler type is specified
                raise ValueError('Invalid lr_scheduler type!')
            
            # Return a list containing the optimizer and the scheduler
            return [optimizer], [scheduler]
    
    
    # Define a function to configure loss functions
    def configure_loss(self):
        # Get the loss from hyperparameters
        loss = self.hparams.loss
        
        # If loss is a list, loop over the elements and create a loss function for each one
        if isinstance(loss, list):
            for l in loss:
                self.loss_function[l.lower()] = (self.get_loss_function(l.lower()))
        # If loss is not a list, create a loss function for it
        else:
            self.loss_function[loss.lower()] = self.get_loss_function(loss.lower())
    
    # Define a function to get a specific loss function based on the loss name
    def get_loss_function(self, loss_name):
        if loss_name == 'l1':
            loss_function = nn.L1Loss()
        elif loss_name == 'ce':
            loss_function = nn.CrossEntropyLoss(reduction='mean')
        elif loss_name == 'kl':
            loss_function = nn.KLDivLoss(reduction='batchmean')
        else:
            # Raise an error if an invalid loss type is specified
            raise ValueError("Invalid Loss Type!")
        
        # Return the loss function
        return loss_function
    
    # Define a function to load the model
    def load_model(self):
        # Get the model name from hyperparameters
        name = self.hparams.model_name
        
        # Convert the model name to CamelCase
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        
        try:
            # Try to import the specified module and get the specified class
            Model = getattr(importlib.import_module('.' + name, package=__package__), camel_name)
        except:
            # Raise an error if the module or class is invalid
            raise ValueError(f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        
        # Instantiate the model with the given hyperparameters
        self.model = self.instancialize(Model)
    
    # Define a function to instantiate a given model with specified arguments

    def instancialize(self, Model, **other_args):
        # Get the arguments of the Model's __init__ method
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        
        # Get the keys of self.hparams
        inkeys = self.hparams.keys()
        
        # Filter out the arguments that are not in self.hparams
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        
        # Update args1 with other_args
        args1.update(other_args)
        
        # Return an instance of Model with the filtered arguments
        return Model(**args1)

    def cal_loss(self, stu_encode, tea_encode, step_name):
        losses = []
        
        # Iterate through each loss function and its corresponding scale
        for (loss_name, loss), scale in zip(self.loss_function.items(), self.hparams.loss_scale):
            if loss_name == 'kl':
                # Compute the KL divergence loss with temperature scaling
                loss_res = loss(
                    F.softmax(stu_encode / self.hparams.t, dim=1).log(),
                    F.softmax(tea_encode / self.hparams.t, dim=1)
                ) * self.hparams.t ** 2
            elif loss_name == 'l1':
                # Compute the L1 loss
                loss_res = loss(stu_encode, tea_encode)
            elif loss_name == 'ce':
                # Compute the cross-entropy loss
                loss_res = loss(
                    stu_encode.softmax(dim=1),
                    tea_encode.softmax(dim=1)
                )
            else:
                # Raise an error if the loss function is not found
                raise ValueError('loss function not found!')
            
            # Scale the loss by its corresponding scale value
            loss_res *= scale
            
            # Log the loss value
            self.log('{}_loss/'.format(step_name) + loss_name, loss_res.item(), on_step=True, on_epoch=True, prog_bar=False)
            
            # Append the scaled loss to the list of losses
            losses.append(loss_res)

        # Compute the total loss by either averaging the losses or weighted summing them
        if self.hparams.weight:
            assert len(self.hparams.weight) == len(
                losses), 'the number of self.weight should be the same as the number of loss'
            assert sum(self.hparams.weight) == 1, 'sum of wight should be 1, instead of {}'.format(
                sum(self.hparams.weight))
            total_loss = sum([loss * weight for loss, weight in zip(losses, self.hparams.weight)])
        else:
            total_loss = sum(losses) / len(losses)
        
        # Log the total loss value
        self.log('{}_loss/total_loss'.format(step_name), total_loss.item(), on_step=True, on_epoch=True, prog_bar=False)
        
        # Return the total loss value
        return total_loss
