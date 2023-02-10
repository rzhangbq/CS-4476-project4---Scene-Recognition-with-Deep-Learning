
import torch
import torch.nn as nn


class SimpleNetFinal(nn.Module):
    def __init__(self):
        """
        Init function to define the layers and loss function

        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means
        """
        super(SimpleNetFinal, self).__init__()

        self.conv_layers = nn.Sequential()
        self.fc_layers = nn.Sequential()
        self.loss_criterion = None

        ############################################################################
        # Student code begin
        ############################################################################
        self.loss_criterion = nn.CrossEntropyLoss()
        self.conv_layers = nn.Sequential(
          nn.Conv2d(1, 10, 5),
          nn.BatchNorm2d(10),
          nn.MaxPool2d(3, 2, 1),
          nn.ReLU(True), # 30*30
          nn.Conv2d(10, 13, 5, padding = 2),
          nn.BatchNorm2d(13),
          nn.Conv2d(13, 17, 5, padding = 2), # 30*30
          nn.BatchNorm2d(17),
          nn.MaxPool2d(3, 2, 1), 
          nn.ReLU(True), # 15*15
          nn.Dropout(0.3),
          nn.Conv2d(17, 20, 5, padding= 2),
          nn.BatchNorm2d(20),
          nn.MaxPool2d(3, 3),
          nn.ReLU(True),
          nn.Flatten(),
        )

        self.fc_layers = nn.Sequential(
          nn.Linear(500, 100),
          nn.Linear(100, 15)
        )


        ############################################################################
        # Student code end
        ############################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass with the net

        Args:
        -   x: the input image [Dim: (N,C,H,W)]
        Returns:
        -   y: the output (raw scores) of the net [Dim: (N,15)]
        """
        model_output = None
        ############################################################################
        # Student code begin
        ############################################################################
        
        model_output = self.fc_layers(self.conv_layers(x))

        
        ############################################################################
        # Student code end
        ############################################################################

        return model_output