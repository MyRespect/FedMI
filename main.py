import os
import numpy as np

import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F

from model import SupConLoss, AudioEncoder, DepthEncoder, RadarEncoder, AutoEncoder, ContrastiveAutoEncoder, Classifier
from dataset import UnimodalDataset, MultimodalDataset, parse_option, set_loader

folder_path = './trial_1/'
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

def train_unimodal_encoder(audio_encoder, depth_encoder, radar_encoder, train_loader, device, epochs=30):

    loss_function = nn.MSELoss().to(device)

    optimizer_1 = optim.SGD(audio_encoder.parameters(), lr=1*1e-1)
    optimizer_2 = optim.SGD(depth_encoder.parameters(), lr=7*1e-1)
    optimizer_3 = optim.SGD(radar_encoder.parameters(), lr=3*1e-1)

    print("******* Training Unimodal Encoder *******")

    if os.path.exists(folder_path+'audio_encoder.pt'):
        audio_encoder = torch.load(folder_path+'audio_encoder.pt')
        depth_encoder = torch.load(folder_path+'depth_encoder.pt')
        radar_encoder = torch.load(folder_path+'radar_encoder.pt')

    audio_encoder.train()
    depth_encoder.train()
    radar_encoder.train()

    for epoch in range(epochs):
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

    torch.save(audio_encoder, folder_path+'audio_encoder.pt')
    torch.save(depth_encoder, folder_path+'depth_encoder.pt')
    torch.save(radar_encoder, folder_path+'radar_encoder.pt')

def train_autoencoder(autoencoder, train_loader, device, encoder_tuple, opt, epochs=50):

    loss_function = nn.MSELoss().to(device)

    optimizer_1 = optim.SGD(autoencoder.encoder_1.parameters(), lr=1*1e-5)
    optimizer_2 = optim.SGD(autoencoder.encoder_2.parameters(), lr=1*1e-4)
    optimizer_3 = optim.SGD(autoencoder.encoder_3.parameters(), lr=1*1e-5)

    audio_encoder, depth_encoder, radar_encoder = encoder_tuple

    print("******* Training AutoEncoder *******")

    if os.path.exists(folder_path+'autoencoder.pt'):
        autoencoder = torch.load(folder_path+'autoencoder.pt')

    autoencoder.train()

    for epoch in range(epochs):

        print("The epoch: ", epoch)

        for idx, (data_pair, labels) in enumerate(train_loader):
            audio_data = data_pair[0].to(device)
            depth_data = data_pair[1].to(device)
            radar_data = data_pair[2].to(device)

            audio_feature = audio_encoder.extract_feature(audio_data)
            depth_feature = depth_encoder.extract_feature(depth_data)
            radar_feature = radar_encoder.extract_feature(radar_data)

            x1_recon, x2_recon, x3_recon = autoencoder(audio_feature, depth_feature, radar_feature)

            audio_feature = audio_feature.view(opt.batch_size, -1)
            depth_feature = depth_feature.view(opt.batch_size, -1)
            radar_feature = radar_feature.view(opt.batch_size, -1)

            loss_1 = loss_function(x1_recon, audio_feature)
            loss_2 = loss_function(x2_recon, depth_feature)
            loss_3 = loss_function(x3_recon, radar_feature)

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

    torch.save(autoencoder, folder_path+'autoencoder.pt')

def train_contrastive(contrastive_model, train_loader, device, encoder_tuple, opt, epochs=5):

    loss_function_1 = nn.MSELoss().to(device)

    loss_function_2 = SupConLoss(device=opt.device)

    optimizer = optim.Adam(contrastive_model.parameters(), lr=0.007)

    audio_encoder, depth_encoder, radar_encoder, autoencoder = encoder_tuple

    print("******* Training Contrastive Model *******")

    if os.path.exists(folder_path+'contrastive_model.pt'):
        contrastive_model = torch.load(folder_path+'contrastive_model.pt')

    contrastive_model.train()

    for epoch in range(epochs):

        print("The epoch: ", epoch)

        for idx, (data_pair, labels) in enumerate(train_loader):
            audio_data = data_pair[0].to(device)
            depth_data = data_pair[1].to(device)
            radar_data = data_pair[2].to(device)

            audio_feature = audio_encoder.extract_feature(audio_data)
            depth_feature = depth_encoder.extract_feature(depth_data)
            radar_feature = radar_encoder.extract_feature(radar_data)

            audio_1, depth_1, radar_1 = autoencoder.extract_feature(audio_feature, depth_feature, radar_feature)

            for feat_item in [audio_1, depth_1, radar_1]:
                x1_recon, x2_recon, x3_recon = contrastive_model(feat_item)

                loss_1 = loss_function_1(x1_recon, audio_1)
                loss_2 = loss_function_1(x2_recon, depth_1)
                loss_3 = loss_function_1(x3_recon, radar_1)

                hidden_feature = torch.stack([x1_recon, x2_recon, x3_recon])
                hidden_feature = hidden_feature.view(opt.batch_size, 3, -1)
                loss_contrastive = loss_function_2(hidden_feature)

                loss_all = loss_1+loss_2+loss_3+loss_contrastive

                optimizer.zero_grad()

                loss_all.backward(retain_graph=True)

                optimizer.step()

    torch.save(contrastive_model, folder_path+'contrastive_model.pt')

def train_classifier(classifier, encoder_tuple, opt, epochs=5):

    loss_func = nn.CrossEntropyLoss().to(device)

    optimizer = optim.Adam(classifier.parameters(), lr=0.0007)

    audio_encoder, depth_encoder, radar_encoder, autoencoder, contrastive_model = encoder_tuple

    print("******* Training Classification Model *******")

    # if os.path.exists(folder_path+'classifier.pt'):
    #     classifier = torch.load(folder_path+'classifier.pt')

    classifier.train()

    acc_list = []
    loss_list = []
    for epoch in range(epochs):
        correct = 0
        batch_cnt = 0
        acc_sum = 0

        for idx, (data_pair, label) in enumerate(train_loader):
            audio_data = data_pair[0].to(device)
            depth_data = data_pair[1].to(device)
            radar_data = data_pair[2].to(device)

            label = label.to(device)

            audio_feature = audio_encoder.extract_feature(audio_data)
            depth_feature = depth_encoder.extract_feature(depth_data)
            radar_feature = radar_encoder.extract_feature(radar_data)

            audio_1, depth_1, radar_1 = autoencoder.extract_feature(audio_feature, depth_feature, radar_feature)

            # h_sum = contrastive_model.extract_feature(audio_1, depth_1, radar_1)

            h_sum = torch.sum(torch.stack([audio_1, depth_1, radar_1]), dim=0)

            output = classifier(h_sum)
            loss = loss_func(output, label)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            pred = output.argmax(dim=1)  # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item()
            batch_cnt += 1
        acc = correct / (len(train_loader.dataset))
        acc_list.append(acc)
        loss_list.append(loss.item())
        if epoch % 1 == 0:
            print('Train Epoch: {} Loss: {:.6f} Accuracy: {:.4f}'.format(epoch, loss.item(), 100. * acc))

    torch.save(classifier, folder_path+'classifier.pt')
    print("Loss: ", loss_list)    


if __name__ == "__main__":

    opt = parse_option()
    opt.usr_id = 16
    opt.local_modality = 'all'
    train_loader, val_loader = set_loader(opt)

    # print("len(train_loader), len(val_loader), len(val_loader.dataset): ", len(train_loader), len(val_loader), len(val_loader.dataset))

    ########################## Unimodal Module ##########################

    train_mode = False

    audio_encoder = AudioEncoder().to(device)
    depth_encoder = DepthEncoder().to(device)
    radar_encoder = RadarEncoder().to(device)

    if train_mode == True:
        train_unimodal_encoder(audio_encoder, depth_encoder, radar_encoder, train_loader, device, epochs=50)

    audio_encoder = torch.load(folder_path+'audio_encoder.pt')
    depth_encoder = torch.load(folder_path+'depth_encoder.pt')
    radar_encoder = torch.load(folder_path+'radar_encoder.pt')

    audio_encoder.eval()
    depth_encoder.eval()
    radar_encoder.eval()

    ########################## Autoencoder Module ##########################

    train_mode = False 
    
    autoencoder = AutoEncoder(batch_size = opt.batch_size, mask_ratio = 0.2).to(device)

    if train_mode == True:
        train_autoencoder(autoencoder, train_loader, device, (audio_encoder, depth_encoder, radar_encoder), opt, epochs=20)

    autoencoder = torch.load(folder_path+'autoencoder.pt')

    autoencoder.eval()

    ########################## Contrastive Autoencoder Module ##########################

    train_mode = False 

    contrastive_model = ContrastiveAutoEncoder().to(device)

    if train_mode == True:
        train_contrastive(contrastive_model, train_loader, device, (audio_encoder, depth_encoder, radar_encoder, autoencoder), opt, epochs=5)

    contrastive_model = torch.load(folder_path+'contrastive_model.pt')

    contrastive_model.eval()

    ########################## Classification Module ##########################

    train_mode = True

    classifier = Classifier().to(device)


    if train_mode == True:
        train_classifier(classifier, (audio_encoder, depth_encoder, radar_encoder, autoencoder, contrastive_model), opt, epochs=100)

    classifier = torch.load(folder_path+'classifier.pt')
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

        audio_1, depth_1, radar_1 = autoencoder.extract_feature(audio_feature, depth_feature, radar_feature)

        h_sum = torch.sum(torch.stack([audio_1, depth_1, radar_1]), dim=0)

        # h_sum = contrastive_model.extract_feature(audio_1, depth_1, radar_1)
        # print(np.shape(h_sum)) # [batch_size, 768]

        # h_sum = h_sum.view(opt.batch_size, 3, -1)

        output = classifier(h_sum)
        pred = output.argmax(dim=1)  # get the index of the max log-probability
        correct += pred.eq(label.view_as(pred)).sum().item()
        batch_cnt += 1
    print('Test Accuracy: {:.4f}'.format(100.*correct / (batch_cnt*batch_size)))
