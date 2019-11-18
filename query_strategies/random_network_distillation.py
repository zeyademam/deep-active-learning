import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .strategy import Strategy
from model import ResNet18, init_params


class RandomNetworkDistillation(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, experiment):
        super(RandomNetworkDistillation, self).__init__(X,
                                                        Y,
                                                        idxs_lb,
                                                        net,
                                                        handler,
                                                        args,
                                                        experiment)
        self.predictor_net = ResNet18(in_channels=3).to(self.device)
        self.predictor_state = self.predictor_net.state_dict()

    def query(self, n):
        # Freeze classifier weights
        for param in self.clf.parameters():
            param.requires_grad = False

        # Train predictor on labeled data with target outputs as labels
        self.predictor_net.apply(init_params)
        self.train_predictor()

        # Run predictor on unlabeled data with target outputs as labels
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        scores = self.score_data(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        # Higher scores have higher loss and therefore are most uncertain
        scores_sorted, idxs = scores.sort(descending=True)

        # Unfreeze classifier weights
        for param in self.clf.parameters():
            param.requires_grad = True

        # Return the n unlabeled samples which received highest uncertainty
        return idxs_unlabeled[idxs[:n]]

    def score_data(self, X, Y):
        """Returns uncertainty scores on giving data.
        """
        self.predictor_state = self.predictor_net.state_dict()
        loader_te = DataLoader(
            self.handler(X, Y, transform=self.args['transform']),
            shuffle=False, **self.args['loader_te_args']
        )

        self.predictor_net.eval()
        scores = torch.zeros(len(Y), dtype=Y.dtype)
        with torch.no_grad():
            for batch_idx, (x, _, idxs) in enumerate(loader_te):
                x = x.to(self.device)
                _, target_embedding = self.clf(x)
                _, predicted_embedding = self.predictor_net(x)
                loss = F.mse_loss(predicted_embedding, target_embedding)
                scores[idxs] = loss.cpu()  # Higher scores are most uncertain!
        return scores

    def train_predictor(self):
        # Train predictor network on already labeled data
        # First round trains for n_epochs, afterwards trains for less epochs
        if self.round == 1:
            n_epoch = self.args['n_epoch']
        else:
            n_epoch = self.args['n_epoch_distill']
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        optimizer = optim.SGD(self.predictor_net.parameters(),
                              **self.args['distill_optimizer_args'])
        loader_tr = DataLoader(
            self.handler(self.X[idxs_train], self.Y[idxs_train],
                         transform=self.args['transform']),
            shuffle=True, **self.args['loader_tr_args']
        )

        for epoch in range(1, n_epoch + 1):
            self._train_predictor(epoch, loader_tr, optimizer)

    def _train_predictor(self, epoch, loader_tr, optimizer):
        self.predictor_net.train()
        for batch_idx, (x, _, idxs) in enumerate(loader_tr):
            x = x.to(self.device)
            optimizer.zero_grad()
            _, target_embedding= self.clf(x)
            _, predicted_embedding = self.predictor_net(x)
            loss = F.mse_loss(predicted_embedding, target_embedding)
            loss.backward()
            optimizer.step()
