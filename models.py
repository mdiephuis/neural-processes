import torch
import torch.nn as nn
import torch.distributions as td
import torch.optim


class SimpleNP(nn.Module):
    def __init__(self, x_dim=1, y_dim=1, h_dim=16, r_dim=32, z_dim=32):
        super(SimpleNP, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.g_dim = h_dim
        self.r_dim = r_dim
        self.z_dim = z_dim

        self.encoder_block = nn.Sequential(
            nn.Linear(self.x_dim + self.y_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.r_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.r_dim, self.r_dim)
        )
        self.decoder_block = nn.Sequential(
            nn.Linear(self.z_dim + self.x_dim, self.g_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.g_dim, self.g_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.g_dim, self.y_dim)
        )

        self.mu_layer = nn.Linear(self.r_dim, self.z_dim)
        self.std_layer = nn.Linear(self.r_dim, self.z_dim)

    def reparameterize(self, r):
        mu = self.mu_layer(r)
        sigma = self.std_layer(r)
        return mu, sigma

    def encode(self, x, y):
        batch_sz, num_points = x.size(0), x.size(1)
        xy = torch.cat((x, y), -1).view(batch_sz * num_points, -1)
        r = self.encoder_block(xy).view(batch_sz, num_points, self.r_dim)
        r = torch.mean(r, dim=1).view(batch_sz, -1)
        return r

    def decode(self, xt, z):
        xz = torch.cat((xt, z), -1)
        return self.decoder_block(xz)

    def forward(self, xc, yc, xt=None, yt=None):
        num_points = xc.size(1)
        if self.training:

            r_context = self.encode(xc, yc)
            r_target = self.encode(xt, yt)

            # q(z | context)
            mu_c, std_c = self.reparameterize(r_context)  # (batch_s x z_dim)
            qz_c = td.normal.Normal(mu_c, std_c)

            # q(z | context, target)
            mu_ct, std_ct = self.reparameterize(r_target)  # (num_points x z_dim)
            qz_ct = td.normal.Normal(mu_ct, std_ct)

            z_draw = qz_ct.rsample()
            z_draw = z_draw.unsqueeze(1).repeat(1, num_points, 1)

            # prob wrong
            yt_hat = self.decode(xt, z_draw)

            return yt_hat, qz_c, qz_ct

        else:
            r_context = self.encode(xc, yc)

            mu, std = self.reparameterize(r_context)
            q = torch.distributions.normal.Normal(mu, std)
            z_draw = q.rsample()

            z_draw = z_draw.unsqueeze(1).repeat(1, num_points, 1)
            yt_hat = self.decode(xt, z_draw)

            return yt_hat


def neural_tangent_kernel_1d(model, x):
    '''
        Determine the neural tangent kernel of the model and inputs
        and the gradient feature map

        1D only
    '''
    output = model(x)
    param_vector = nn.utils.parameters_to_vector(model.parameters())

    # transpose jacobian: (grad y(w))^T)
    features = torch.zeros(output.size(0), param_vector.size(0), requires_grad=False)

    # Loop over all data points
    for i in range(output.size(0)):
        model.zero_grad()
        output[i].backward(retain_graph=True)
        all_grad = []
        for p in model.parameters():
            all_grad.append(p.grad.reshape(-1))
        features[i, :] = torch.cat(all_grad)

    # tangent_kernel = torch.bmm(features, features.t())
    tangent_kernel = features @ features.t()

    return features, tangent_kernel


def gradient_descent(model, x, y, n_iters=100):
    '''
    Run gradient descent using square loss on the model and all the data
    '''
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_func = nn.MSELoss()
    model.train()

    for i in range(n_iters):
        y_hat = model(x)

        loss = loss_func(y_hat, y)

        model.zero_grad()
        loss.backward()
        optim.step()

    return

