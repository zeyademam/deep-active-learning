import numpy as np
import time
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
        print("Starting Predictor Training on labeled data.")
        self.train_predictor()
        print("Finished training Predictor on labeled data.")

        # Run predictor on unlabeled data with target outputs as labels
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        # Split the unlabeled 85% train/ 20% holdout test randomly
        k = int(float(len(idxs_unlabeled)) * .85)
        idxs_tmp = np.arange(len(idxs_unlabeled))
        np.random.shuffle(idxs_tmp)
        idxs_rnd_train = idxs_unlabeled[idxs_tmp[:k]]
        idxs_rnd_test = idxs_unlabeled[idxs_tmp[k:]]

        scores = self.score_data(self.X[idxs_rnd_train],
                                 self.X[idxs_rnd_test])
        # Lower scores have lower loss so that sample was more representative
        print("Started querying.")
        scores_sorted, idxs = scores.sort(descending=False)

        # Unfreeze classifier weights
        for param in self.clf.parameters():
            param.requires_grad = True

        # Return the n unlabeled samples which received highest uncertainty
        return idxs_rnd_train[idxs[:n]]

    def score_data(self, X_train, X_test):
        """Returns uncertainty scores on giving data.
        """
        # Save the predictor's state
        self.predictor_state = self.predictor_net.state_dict()

        # Define optimizer and init scores array (output)
        optimizer = optim.SGD(self.predictor_net.parameters(),
                              **self.args['distill_optimizer_args'])
        scores = torch.zeros(len(X_train))
        loader_tr = DataLoader(
            self.handler(X_train, np.zeros(len(X_train)),
                         transform=self.args['transform']),
            shuffle=True, batch_size=1, drop_last=False)
        loader_te = DataLoader(
            self.handler(X_test, np.zeros(len(X_test)),
                         transform=self.args['transform']),
            shuffle=True, batch_size=5000, drop_last=False)
        # Iterate over train data
        for count, (x, _, idx) in enumerate(loader_tr):
            start = time.time()
            # Restore predictor original state
            self.predictor_net.load_state_dict(self.predictor_state)
            # Perform two SGD passes on current data point
            optimizer.zero_grad()
            for _ in range(2):
                x = x.to(self.device)
                _, target_embedding = self.clf(x)
                _, predicted_embedding = self.predictor_net(x)
                loss = F.mse_loss(predicted_embedding, target_embedding)
                loss.backward()
                optimizer.step()

            self.predictor_net.eval()
            avg_loss = 0
            # Evaluate loss on "test data"
            with torch.no_grad():
                for t, _, _ in loader_te:
                    t = t.to(self.device)
                    _, target_embedding = self.clf(t)
                    _, predicted_embedding = self.predictor_net(t)
                    avg_loss += F.mse_loss(predicted_embedding,
                                           target_embedding).cpu()
                avg_loss /= len(X_test)

            scores[idx] = avg_loss  # Lower scores are better
            end = time.time()
            print(f"Scored element {count+1}/{len(loader_tr)} "
                  f"in {end-start:.2f} seconds.")
        return scores

    def train_predictor(self):
        # Train predictor network on already labeled data
        # First round trains for n_epochs, afterwards trains for less epochs
        n_epoch = self.args['n_epoch']

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        optimizer = optim.SGD(self.predictor_net.parameters(),
                              **self.args['distill_optimizer_args'])
        loader_tr = DataLoader(
            self.handler(self.X[idxs_train], self.Y[idxs_train],
                         transform=self.args['transform']),
            shuffle=True, **self.args['loader_tr_args']
        )

        for epoch in range(1, n_epoch + 1):
            optimizer.zero_grad()
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
