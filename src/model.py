import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        self.model = nn.Sequential(
            
            # first conv layer + batchnorm + relu + maxpool
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1), #224x224x3 -> 224x224x16 
#             nn.BatchNorm2d(16), # not to batchnorm raw-data (the input data to your model). but only subsequent layers
            nn.MaxPool2d(2, 2),# -> maxpool 112x112x16
            nn.ReLU(),
#             nn.Dropout2d(dropout),
            
            # second conv layer + batchnorm + relu + maxpool
            nn.Conv2d(16, 32, 3, padding=1),  #112x112x16 -> 112x112x32
            # Add batch normalization (BatchNorm2d) here
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),  # -> maxpool 56x56x32
            nn.ReLU(),
#             nn.Dropout2d(dropout),
            
            # third conv layer + batchnorm + relu + maxpool
            nn.Conv2d(32, 64, 3, padding=1),  # 56x56x32 -> 56x56x64 
            # Add batch normalization (BatchNorm2d) here
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),  # -> maxpool 28x28x64
            nn.ReLU(),
#             nn.Dropout2d(dropout),
            
            # Since we are using BatchNorm and data augmentation,
            # we can go deeper than before and add one more conv layer
            # forth conv layer + batchnorm + relu + maxpool
            nn.Conv2d(64, 128, 3, padding=1),  # 28x28x64 -> 28x28x128 
            # Add batch normalization (BatchNorm2d) here
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),  # -> maxpool 14x14x128
            nn.ReLU(),
#             nn.Dropout2d(dropout),
            
            # fifth conv layer + batchnorm + relu + maxpool
            nn.Conv2d(128, 256, 3, padding=1),  # 14x14x128 -> 14x14x256
            # Add batch normalization (BatchNorm2d) here
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> maxpool 7x7x256
#             nn.Dropout2d(dropout),
            
            nn.Flatten(),  # -> 1x7x7x256
            
            nn.Linear(7 * 7 * 256, 512),  # -> 512
            nn.Linear(512, 128),
            nn.Dropout(dropout),
            # Add batch normalization (BatchNorm1d, NOT BatchNorm2d) here
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        x = self.model(x)
        return x

    
######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
