import numpy as np 
# np.seterr(invalid='ignore')

import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

from dataset import UnimodalDataset, MultimodalDataset, set_loader, parse_option

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda:3')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # unbind, remove a certain dimension and return a tuple of sliced tensor.
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # print("anchor_dot_contrast", anchor_dot_contrast)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits_min, _ = torch.min(anchor_dot_contrast, dim=1, keepdim=True)
        logits = (anchor_dot_contrast - logits_max.detach())/(logits_min.detach()-logits_max.detach())

        # tile mask!!!
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

# audio input: [bsz, 20, 87]
class AudioEncoder(nn.Module):
    def __init__(self, in_channel=20, out_channel=128, kernel_size=4, stride=1, padding=0):
        super(AudioEncoder, self).__init__()
        self.encoder = nn.Sequential(
            # nn.BatchNorm1d(in_channel),
            nn.Conv1d(in_channel, 32, kernel_size, stride, padding),
            nn.Conv1d(32, 64, kernel_size, stride, padding),
            nn.Conv1d(64, out_channel, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channel)
            )
        self.pooling = nn.MaxPool1d(kernel_size, stride, padding, return_indices=True)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(out_channel, 64, kernel_size, stride, padding),
            nn.ConvTranspose1d(64, 32, kernel_size, stride, padding),
            nn.ConvTranspose1d(32, in_channel, kernel_size, stride, padding),
            )
        self.unpooling = nn.MaxUnpool1d(kernel_size, stride, padding)

    def forward(self, x):
        x = self.encoder(x)
        # print(np.shape(x)) # [16, 128, 78]
        x, indices = self.pooling(x)
        x = self.unpooling(x, indices) # filter out part of information since non-maximal values are lost
        x = self.decoder(x)
        return x

    def extract_feature(self, x_data):
        out = self.encoder(x_data)
        return out

# depth input: [bsz, 16, 112, 112]
class DepthEncoder(nn.Module):
    def __init__(self, in_channel=16, out_channel=128, kernel_size=(3, 3), stride=(3, 3), padding=(1, 1)):
        super(DepthEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size, stride, padding),
            nn.Conv2d(32, 64, kernel_size, 1, padding),
            nn.Conv2d(64, out_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channel)
            )
        self.pooling = nn.MaxPool2d(kernel_size, stride, padding, return_indices=True)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(out_channel, 64, kernel_size, stride, padding),
            nn.ConvTranspose2d(64, 32, kernel_size, 1, padding),
            nn.ConvTranspose2d(32, in_channel, kernel_size, stride, output_padding=(1,1)),
            )
        self.unpooling = nn.MaxUnpool2d(kernel_size, stride, padding)

    def forward(self, x):
        x = self.encoder(x)
        # print(np.shape(x)) # [16, 128, 13, 13]
        x, indices = self.pooling(x)
        x = self.unpooling(x, indices) # filter out part of information since non-maximal values are lost
        x = self.decoder(x)
        return x

    def extract_feature(self, x_data):
        out = self.encoder(x_data)
        return out

# radar input: [bsz, 20, 2, 16, 32, 16]
class RadarEncoder(nn.Module):
    def __init__(self, in_channel=40, out_channel=128, kernel_size=(2, 2, 2), stride=(1, 1, 1), padding=0):
        super(RadarEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channel, 32, kernel_size, (2, 2, 2), padding),
            nn.Conv3d(32, 64, kernel_size, stride, padding),
            nn.Conv3d(64, out_channel, kernel_size, stride, padding),
            nn.BatchNorm3d(out_channel)
            )
        self.pooling = nn.MaxPool3d(kernel_size, stride, padding, return_indices=True)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(out_channel, 64, kernel_size, stride, padding),
            nn.ConvTranspose3d(64, 32, kernel_size, stride, padding),
            nn.ConvTranspose3d(32, in_channel, kernel_size, (2, 2, 2), padding),
            )
        self.unpooling = nn.MaxUnpool3d(kernel_size, stride, padding)


    def forward(self, x):
        x = x.view(-1, 40, 16, 32, 16)
        x = self.encoder(x)
        # print(np.shape(x)) # [16, 128, 6, 14, 6]
        x, indices = self.pooling(x)
        x = self.unpooling(x, indices) # filter out part of information since non-maximal values are lost
        x = self.decoder(x)
        x = x.view(-1, 20, 2, 16, 32, 16)
        return x

    def extract_feature(self, x_data):
        x_data = x_data.view(-1, 40, 16, 32, 16)
        out = self.encoder(x_data)
        return out

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder_1 = nn.Sequential(
            nn.Linear(128*78, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
            )
        self.encoder_2 = nn.Sequential(
            nn.Linear(128*13*13, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
            )
        self.encoder_3 = nn.Sequential(
            nn.Linear(128*6*14*6, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
            )
        self.decoder_1 = nn.Sequential(
            nn.Linear(256*1, 512),
            nn.ReLU(),
            nn.Linear(512, 128*78)
            )
        self.decoder_2 = nn.Sequential(
            nn.Linear(256*1, 512),
            nn.ReLU(),
            nn.Linear(512, 128*13*13)
            )
        self.decoder_3 = nn.Sequential(
            nn.Linear(256*1, 512),
            nn.ReLU(),
            nn.Linear(512, 128*6*14*6)
            )

    def forward(self, x_1, x_2, x_3):
        # h_1 = self.encoder_1(x_1)
        # h_2 = self.encoder_2(x_2)
        h_3 = self.encoder_3(x_3)
        # h_sum = (h_1+h_2+h_3)

        # h_sum = torch.cat((h_2, h_3), dim=1)
        h_sum = h_3

        # print(np.shape(h_sum), np.shape(h_1))

        # x1_recon = self.decoder_1(h_sum)
        # x2_recon = self.decoder_2(h_sum)
        x3_recon = self.decoder_3(h_sum)
        return 0, 0, x3_recon

    def extract_feature(self, x_1, x_2, x_3):
        # h_1 = self.encoder_1(x_1)
        # h_2 = self.encoder_2(x_2)
        h_3 = self.encoder_3(x_3)

        # h_sum = torch.cat((h_1, h_2, h_3), dim=1)
        # h_sum = torch.cat((h_2, h_3), dim=1)
        h_sum = h_3

        return h_sum

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv = nn.Conv1d(1, 8, 2)
        self.fc1 = nn.Linear(255*8, 256*4)  # note to change the parameter
        self.fc2 = nn.Linear(256*4, 256*2)  # note to change the parameter
        self.fc3 = nn.Linear(256*2, 256)  # note to change the parameter
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 11)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        # x = F.relu(x)
        x = self.fc4(x)
        # x = F.relu(x)
        x = self.fc5(x)
        # x = F.relu(x)
        x = self.fc6(x)
        return x

if __name__ == "__main__":

    train_mode = False

    audio_encoder = AudioEncoder().to(device)
    depth_encoder = DepthEncoder().to(device)
    radar_encoder = RadarEncoder().to(device)

    opt = parse_option()

    opt.usr_id = 16
    opt.local_modality = 'all'
    train_loader, val_loader = set_loader(opt)

    print("len(train_loader)", len(train_loader), len(val_loader), len(val_loader.dataset))

    loss_function = nn.MSELoss().to(device)
    optimizer_1 = optim.SGD(audio_encoder.parameters(), lr=1*1e-1)
    optimizer_2 = optim.SGD(depth_encoder.parameters(), lr=7*1e-1)
    optimizer_3 = optim.SGD(radar_encoder.parameters(), lr=3*1e-1)

    contrastive_loss_func = SupConLoss() # we use the unsupervised version

    if train_mode == True:
        print("******* Unimodal AutoEncoder *******")
        audio_encoder = torch.load('audio_encoder.pt')
        depth_encoder = torch.load('depth_encoder.pt')
        radar_encoder = torch.load('radar_encoder.pt')
        audio_encoder.train()
        depth_encoder.train()
        radar_encoder.train()
        for epoch in range(30):
            print("The epoch: ", epoch)
            for idx, (data_pair, labels) in enumerate(train_loader):
                audio_data = data_pair[0].to(device)
                depth_data = data_pair[1].to(device)
                radar_data = data_pair[2].to(device)

                audio_output = audio_encoder(audio_data)
                depth_output = depth_encoder(depth_data)
                radar_output = radar_encoder(radar_data)

                loss_1 = loss_function(audio_output, audio_data)
                loss_2 = loss_function(depth_output, depth_data)
                loss_3 = loss_function(radar_output, radar_data)

                optimizer_1.zero_grad()
                optimizer_2.zero_grad()
                optimizer_3.zero_grad()

                loss_1.backward(retain_graph=True)
                loss_2.backward(retain_graph=True)
                loss_3.backward(retain_graph=True)

                optimizer_1.step()
                optimizer_2.step()
                optimizer_3.step()

            print("Loss: ", loss_1.item(), loss_2.item(), loss_3.item())

        torch.save(audio_encoder, 'audio_encoder.pt')
        torch.save(depth_encoder, 'depth_encoder.pt')
        torch.save(radar_encoder, 'radar_encoder.pt')
    else:
        audio_encoder = torch.load('audio_encoder.pt')
        depth_encoder = torch.load('depth_encoder.pt')
        radar_encoder = torch.load('radar_encoder.pt')

    train_mode = True

    autoencoder = AutoEncoder().to(device)
    optimizer_4 = optim.SGD(autoencoder.parameters(), lr=7*1e-1)

    if train_mode == True:
        print("******* AutoEncoder *******")
        # autoencoder = torch.load('autoencoder.pt')
        autoencoder.train()
        for epoch in range(10):
            print("The epoch: ", epoch)
            for idx, (data_pair, labels) in enumerate(train_loader):
                audio_data = data_pair[0].to(device)
                depth_data = data_pair[1].to(device)
                radar_data = data_pair[2].to(device)

                audio_feature = audio_encoder.extract_feature(audio_data)
                depth_feature = depth_encoder.extract_feature(depth_data)
                radar_feature = radar_encoder.extract_feature(radar_data)

                audio_feature = audio_feature.view(opt.batch_size, -1)
                depth_feature = depth_feature.view(opt.batch_size, -1)
                radar_feature = radar_feature.view(opt.batch_size, -1)

                # modality_feature = torch.vstack((audio_feature[:, :512], depth_feature[:,:512], radar_feature[:,:512])).view(16, 3, -1)

                # con_loss = contrastive_loss_func(modality_feature)

                # audio_feature = torch.rand(16, 9984).to(device)

                # print(np.shape(audio_feature))

                _, _, x3_recon = autoencoder(audio_feature, depth_feature, radar_feature)

                # loss_1 = loss_function(x1_recon, audio_feature)
                # loss_2 = loss_function(x2_recon, depth_feature)
                loss_3 = loss_function(x3_recon, radar_feature)

                # loss = 0.4*loss_1+0.4*loss_2+0.2*loss_3
                loss = loss_3

                # optimizer_1.zero_grad()
                # optimizer_2.zero_grad()
                # optimizer_3.zero_grad()
                # con_loss.backward(retain_graph=True)
                # optimizer_1.step()
                # optimizer_2.step()
                # optimizer_3.step()

                optimizer_4.zero_grad()
                loss.backward(retain_graph=True)
                optimizer_4.step()
            print("Loss: ", loss_3.item())

        torch.save(autoencoder, 'autoencoder.pt')
    else:
        autoencoder = torch.load('autoencoder.pt')

    classifier = Classifier().to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=0.0007)
    loss_func = nn.CrossEntropyLoss().to(device)
    classifier.train()
    batch_size = opt.batch_size

    train_mode = True

    if train_mode == True:
        # print("******* Train Classifier *******")
        # classifier = torch.load('classifier.pt')
        classifier.train()
        for epoch in range(200):
            correct = 0
            batch_cnt = 0
            print("Epoch: ", epoch)
            for idx, (data_pair, label) in enumerate(train_loader):
                audio_data = data_pair[0].to(device)
                depth_data = data_pair[1].to(device)
                radar_data = data_pair[2].to(device)

                label = label.to(device)

                audio_feature = audio_encoder.extract_feature(audio_data)
                depth_feature = depth_encoder.extract_feature(depth_data)
                radar_feature = radar_encoder.extract_feature(radar_data)

                audio_feature = audio_feature.view(opt.batch_size, -1)
                depth_feature = depth_feature.view(opt.batch_size, -1)
                radar_feature = radar_feature.view(opt.batch_size, -1)

                h_sum = autoencoder.extract_feature(audio_feature, depth_feature, radar_feature)

                h_sum = h_sum.view(opt.batch_size, 1, -1)

                classifier.zero_grad()
                output = classifier(h_sum)
                loss = loss_func(output, label)
                loss.backward(retain_graph=True)
                optimizer.step()

                pred = output.argmax(dim=1)  # get the index of the max log-probability
                correct += pred.eq(label.view_as(pred)).sum().item()
                batch_cnt += 1
            if epoch % 2 == 0:
                print('Train Epoch: {} Loss: {:.6f} Accuracy: {:.4f}'.format(epoch, loss.item(), 100. * correct / (batch_cnt*batch_size)))

        torch.save(classifier, 'classifier_2_1.pt')

    classifier = torch.load('classifier_2_1.pt')
    classifier.eval()

    correct = 0
    batch_cnt = 0
    batch_size = opt.batch_size
    print("******* Test Classifier *******")
    for idx, (data_pair, label) in enumerate(val_loader):
        audio_data = data_pair[0].to(device)
        depth_data = data_pair[1].to(device)
        radar_data = data_pair[2].to(device)

        label = label.to(device)

        audio_feature = audio_encoder.extract_feature(audio_data)
        depth_feature = depth_encoder.extract_feature(depth_data)
        radar_feature = radar_encoder.extract_feature(radar_data)

        audio_feature = audio_feature.view(opt.batch_size, -1)
        depth_feature = depth_feature.view(opt.batch_size, -1)
        radar_feature = radar_feature.view(opt.batch_size, -1)

        h_sum = autoencoder.extract_feature(audio_feature, depth_feature, radar_feature)
        print(np.shape(h_sum)) # [batch_size, 768]
        # h_sum[0:255]=0 # acc: 12.38
        h_sum = h_sum.view(opt.batch_size, 1, -1)

        output = classifier(h_sum)
        pred = output.argmax(dim=1)  # get the index of the max log-probability
        correct += pred.eq(label.view_as(pred)).sum().item()
        batch_cnt += 1
    print('Test Accuracy: {:.4f}'.format(100.*correct / (batch_cnt*batch_size)))


