import time

import torch
import torch.nn as nn
from utils import transform_train, transform_val
from torch.optim import lr_scheduler
from model import Policy_Net
from data_generator import data_generator
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


episodes = 5
d_max = 5
n_epochs = 200
iters = 64
batch_size = 32


def train(epoch, model, iters=32):
    total = 0
    acc = 0.0
    model.train()
    for i in range(iters):
        data, label = data_generator(episodes=episodes, batch_size=batch_size, d_max=d_max, transforms=transform_train, shuffle=True)
    #     output = model(data, label)
    #     loss = loss_func(output.permute(1, 2, 0), label)
    #     _, pre = torch.max(output.data, -1)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     scheduler.step()
    #     acc += (pre.view(-1) == label.view(-1)).sum().item()
    #     total += label.view(-1).size(0)
    # print("train   Epoch: {0}/{1} , loss：{2},  acc:{3:.2f}".format(epoch + 1, n_epochs, loss.item(), acc/total))


def val(epoch, model, iters=16):
    total = 0
    acc = 0.0
    model.eval()
    with torch.no_grad():
        for i in range(iters):
            data, label = data_generator(episodes=episodes, transforms=transform_val, shuffle=False)
            # label = label.to(device)
            # data = data.to(device)
    #         output = model(data,label)
    #         _, pre = torch.max(output.data, -1)
    #         loss = loss_func(output, label.squeeze())
    #         acc += (pre.view(-1) == label.view(-1)).sum().item()
    #         total += label.view(-1).size(0)
    #
    # acc = acc/total
    # print("val     Epoch: {0}/{1} , loss：{2},  acc:{3:.2f}".format(epoch + 1, n_epochs, loss.item(), acc))
    return acc


def main():
    policy = Policy_Net().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-04)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters * n_epochs)
    loss_func = nn.CrossEntropyLoss()
    best_val = 0
    start_time = time.time()
    for epoch in range(n_epochs):
        train(epoch, policy)
        # scheduler.step()
        print(time.time() - start_time)
        if (epoch + 1) % 20 == 0:
            acc = val(epoch, policy)
            torch.save(policy, 'model.pth')
            if acc >= best_val:
                model_path = f'model_{epoch}.pth'
                torch.save(policy, model_path)

if __name__ == '__main__':
    main()



