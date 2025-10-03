import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets



def train(net, trainloader, optimizer, scheduler, criterion):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # split data into the image and its label
        inputs, labels = data
        inputs = inputs.to(torch.device)
        labels = labels.to(torch.device)

        # initialise the optimiser
        optimizer.zero_grad()

        # forward
        outputs = net(inputs)
        # backward
        loss = criterion(outputs, labels)
        loss.backward()
        # update the optimizer
        optimizer.step()
        # scheduler.step()
        # loss
        running_loss += loss.item()
    return running_loss

def list_weight_net(net):
    weight=[]
    for name, parameters in net.named_parameters():
        if 'weight' in name and 'features' in name:
            weight.append(name[:-7])
    return weight

def size_of_permutation(net,weight_name):
    for name, parameters in net.named_parameters():
        if weight_name in name:
            return parameters.size()[0]


def permutation_channel(net, weight_name, permut):
    with torch.no_grad():
        for name, parameters in net.named_parameters():
            if weight_name in name:
                if 'weight' in name:
                    print("permutation channel", name)
                    # print(parameters)
                    parameters.copy_(parameters.clone()[:, permut])


def permutation_neuron(net,weight_name, permut):
    with torch.no_grad():
        for name, parameters in net.named_parameters():
            if weight_name in name:
                if 'weight' in name:
                    print("permutation neuron", name)
                    parameters.copy_(parameters.clone()[permut, :])
                else:
                    print("permutation neuron bias", name)
                    parameters.copy_(parameters.clone()[permut])


def permutation(net,permute, layer_n, layer_c):
    permutation_neuron(net,layer_n, permute)
    permutation_channel(net,layer_c, permute)
    return permute

def permutation_bn(net,permut,layer_bn):
    with torch.no_grad():
        for name, parameters in net.named_parameters():
            if layer_bn in name:
                print("permutation batch norm", name)
                # print(parameters)
                print(parameters.running_mean)
                parameters.copy_(parameters.clone()[permut])

def permutation_neuron_bn(net,weight_name, weight_bn, permut):
    with torch.no_grad():
        for name, parameters in net.named_parameters():
            if weight_bn in name:
                print("permutation batch norm",name)
                # print(parameters)
                # print(parameters.running_mean)
                parameters.copy_(parameters.clone()[permut])
            if weight_name in name and 'weight' in name:
                print("permutation neuron",name)
                # print(parameters)
                parameters.copy_(parameters.clone()[permut, :])

def permutation_with_bn(net,permute, layer_n, layer_bn, layer_c):
    permutation_neuron_bn(net,layer_n,layer_bn, permute)
    permutation_channel(net,layer_c, permute)
