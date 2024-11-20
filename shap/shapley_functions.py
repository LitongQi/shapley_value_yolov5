import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.optim import SGD
from scipy.special import comb


def monte_carlo_shapley_value(masks, values):
    N, H, W = masks.shape
    M = H*W
    masks = masks.view(N, M)
    shapley_value = torch.zeros(M)
    for i in range(M):
        mask = masks[:, i]
        mean_value_with_i = values[mask == 1].mean()
        mean_value_wo_i = values[mask == 0].mean()
        shapley_value[i] = mean_value_with_i - mean_value_wo_i

    shapley_value = shapley_value.view(H, W)

    return shapley_value

def kernel_shapley_additive_value(masks, values, use_kernel=True, batch_size=1024, epochs=2000, lr=1e-3):
    N, H, W = masks.shape
    M = H*W
    masks = masks.view(N, M)
    masks = torch.cat([torch.ones(N, 1), masks], dim=1)
    values = values.view(N)

    if use_kernel:
        K = masks.sum(dim=1)
        Kmax = K.max() + 1
        pi = (Kmax - 1) / (K * (Kmax - K)) / comb(Kmax, K)
    else:
        pi = torch.ones_like(values)

    net = nn.Sequential(nn.Linear(M + 1, 1))
    net.cuda()
    loss = nn.MSELoss(reduction="none")
    optimizer = SGD(net.parameters(), lr=lr)

    dataset = Data.TensorDataset(masks.cuda(), values.cuda(), pi.cuda())
    dataloader = Data.DataLoader(dataset, batch_size, shuffle=True)

    for epoch in range(epochs):
        for X, y, w in dataloader:
            output = net(X.float())
            l = loss(output, y.view(-1, 1))
            w = w / w.mean()
            l = (l * w).mean()
            l = l.mean()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        if epoch % 200 == 0:
            print('epoch %d, loss: %f' % (epoch, l.item()))

    shapley_value = net[0].weight.reshape(-1)[1:].reshape(H, W)

    return shapley_value.data.cpu()

def shapley_additive_value(*args, **kwargs):
    return kernel_shapley_additive_value(*args, use_kernel=False, **kwargs)
