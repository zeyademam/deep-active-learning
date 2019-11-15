import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .strategy import Strategy
from model import EmbeddingNet, init_params


class RandomNetworkDistillation(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(RandomNetworkDistillation, self).__init__(X,
                                                        Y,
                                                        idxs_lb,
                                                        net,
                                                        handler,
                                                        args)
        self.target_net = EmbeddingNet().to(self.device)
        self.predictor_net = EmbeddingNet().to(self.device)
        self.target_net.apply(init_params)
        self.predictor_net.apply(init_params)
        self.target_net.eval() # Sets the target network to eval mode

        # Make sure parameters of target model are not updated during training
        for p in self.target_net.parameters():
            p.requires_grad = False

    def query(self, n):
        # TODO: Consider resetting weights of target and predictor nets

        # Train predictor on labeled data with target outputs as labels
        self.train_predictor()

        # Run predictor on unlabeled data with target outputs as labels
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        scores = self.score_data(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        # Higher scores have higher loss and therefore are most uncertain
        scores_sorted, idxs = scores.sort(descending=True)

        # Return the n unlabeled samples which received highest uncertainty
        return idxs_unlabeled[idxs[:n]]

    def score_data(self, X, Y):
        """Returns uncertainty scores on giving data.

        :param X:
        :param Y:
        :return:
        """
        loader_te = DataLoader(
            self.handler(X, Y, transform=self.args['transform']),
            shuffle=False, **self.args['loader_te_args']
        )

        self.predictor_net.eval()
        scores = torch.zeros(len(Y), dtype=Y.dtype)
        with torch.no_grad():
            for batch_idx, (x, _, idxs) in enumerate(loader_te):
                x = x.to(self.device)
                predicition = self.predictor_net(x)
                target = self.target_net(x)
                loss = F.mse_loss(predicition, target)
                scores[idxs] = loss.cpu()  # Higher scores are most uncertain!
        return scores

    def train_predictor(self):
        # Train predictor network on already labeled data
        # First round tains for n_epochs, afterwards trains for less epochs
        if self.round == 1:
            n_epoch = self.args['n_epoch']
        else:
            n_epoch = self.args['n_epoch_distill']
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        optimizer = optim.Adam(self.predictor_net.parameters(),
                               **self.args['distill_optimizer_args'])
        loader_tr = DataLoader(
            self.handler(self.X[idxs_train], self.Y[idxs_train],
                         transform=self.args['transform']),
            shuffle=True, **self.args['loader_tr_args']
        )

        for epoch in range(1, n_epoch + 1):
            self._train(epoch, loader_tr, optimizer)

    def _train_predictor(self, epoch, loader_tr, optimizer):
        self.predictor_net.train()
        for batch_idx, (x, _, idxs) in enumerate(loader_tr):
            x = x.to(self.device)
            optimizer.zero_grad()
            predicition = self.predictor_net(x)
            target = self.target_net(x)
            loss = F.mse_loss(predicition, target)
            loss.backward()
            optimizer.step()
