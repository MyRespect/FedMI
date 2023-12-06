import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # Input: [batch_size, 1, 3, 100], Output: [batch_size, 16, 3, 100]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Input: [batch_size, 16, 3, 100], Output: [batch_size, 32, 3, 100]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Input: [batch_size, 32, 3, 100], Output: [batch_size, 64, 3, 100]
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # Input: [batch_size, 64, 3, 100], Output: [batch_size, 128, 3, 100]
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),  # Input: [batch_size, 128, 3, 100], Output: [batch_size, 64, 3, 100]
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),  # Input: [batch_size, 64, 3, 100], Output: [batch_size, 32, 3, 100]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),  # Input: [batch_size, 32, 3, 100], Output: [batch_size, 16, 3, 100]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1),  # Input: [batch_size, 16, 3, 100], Output: [batch_size, 1, 3, 100]
            nn.ReLU()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Create an instance of the Autoencoder model
model = Autoencoder()

torch.save(model, 'model_test.pt')

# Print the model summary
print(model)