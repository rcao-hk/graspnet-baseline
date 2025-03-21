import numpy as np
import torch
import torch.nn as nn

from abc import ABC, abstractmethod

### Mean-Shift Clustering (PyTorch) ###

def euclidean_distances(x, y):
    """ Computes pairwise distances

        @param x: a [n x d] torch.FloatTensor of datapoints
        @param y: a [m x d] torch.FloatTensor of datapoints

        @return: a [n x m] torch.FloatTensor of pairwise distances
    """
    return torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=2)


def gaussian_kernel(x, y, sigma):
    """ Computes pairwise Gaussian kernel (without normalizing constant)
        (note this is kernel as defined in non-parametric statistics, not a kernel as in RKHS)

        @param x: a [n x d] torch.FloatTensor of datapoints
        @param y: a [m x d] torch.FloatTensor of datapoints
        @param sigma: Gaussian kernel bandwith.
                      Either a scalar, or a [1 x m] torch.FloatTensor of datapoints

        @return: a [n x m] torch.FloatTensor of pairwise kernel computations,
                 without normalizing constant
    """
    return torch.exp(- .5 / (sigma ** 2) * euclidean_distances(x, y) ** 2)


class MeanShift(ABC):
    """ Base abstract class for Mean Shift algorithms w/ diff kernels
    """

    def __init__(self, num_seeds=100, max_iters=10, epsilon=1e-2,
                 h=1., batch_size=None):
        self.num_seeds = num_seeds
        self.max_iters = max_iters
        self.epsilon = epsilon  # connect components parameter
        self.h = h  # kernel bandwidth parameter
        if batch_size is None:
            batch_size = 1000
        self.batch_size = batch_size

        # This should be a function that computes distances w/ func signature: (x,y)
        self.distance = None

        # This should be a function that computes a kernel w/ func signature: (x,y, h)
        self.kernel = None

    def connected_components(self, Z):
        """ Compute simple connected components algorithm.
            @param Z: a [n x d] torch.FloatTensor of datapoints
            @return: a [n] torch.LongTensor of cluster labels
        """

        n, d = Z.shape
        K = 0

        # SAMPLING/GROUPING
        cluster_labels = torch.ones((n,), dtype=torch.long, device=Z.device) * -1
        for i in range(n):
            if cluster_labels[i] == -1:

                # Find all points close to it and label it the same
                distances = self.distance(Z, Z[i:i + 1])  # Shape: [n x 1]
                component_seeds = distances[:, 0] <= self.epsilon

                # If at least one component already has a label, then use the mode of the label
                if torch.unique(cluster_labels[component_seeds]).shape[0] > 1:
                    temp = cluster_labels[component_seeds]
                    temp = temp[temp != -1]
                    label = torch.mode(temp)[0]
                else:
                    label = torch.tensor(K).cuda()
                    K += 1  # Increment number of clusters
                cluster_labels[component_seeds] = label#.cpu()

        return cluster_labels
        # return torch.from_numpy(cluster_labels)

    def seed_hill_climbing(self, X, Z):
        """ Run mean shift hill climbing algorithm on the seeds.
            The seeds climb the distribution given by the KDE of X
            @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
            @param Z: a [m x d] torch.FloatTensor of seeds to run mean shift from
        """

        n, d = X.shape
        m = Z.shape[0]

        for _iter in range(self.max_iters):

            # Create a new object for Z
            new_Z = Z.clone()

            # Compute the update in batches
            for i in range(0, m, self.batch_size):
                W = self.kernel(Z[i:i + self.batch_size], X, self.h)  # Shape: [batch_size x n]
                Q = W / W.sum(dim=1, keepdim=True)  # Shape: [batch_size x n]
                new_Z[i:i + self.batch_size] = torch.mm(Q, X)

            Z = new_Z

        return Z

    def select_smart_seeds(self, X):
        """ Randomly select seeds that far away
            @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
            @return: a [num_seeds x d] matrix of seeds
        """
        n, d = X.shape

        selected_indices = -1 * torch.ones(self.num_seeds, dtype=torch.long)

        # Initialize seeds matrix
        seeds = torch.empty((self.num_seeds, d), device=X.device)
        num_chosen_seeds = 0

        # Keep track of distances
        distances = torch.empty((n, self.num_seeds), device=X.device)

        # Select first seed
        selected_seed_index = np.random.randint(0, n)
        selected_indices[0] = selected_seed_index
        selected_seed = X[selected_seed_index, :]
        seeds[0, :] = selected_seed

        distances[:, 0] = self.distance(X, selected_seed.unsqueeze(0))[:, 0]
        num_chosen_seeds += 1

        # Select rest of seeds
        for i in range(num_chosen_seeds, min(self.num_seeds, n)):
            # Find the point that has the furthest distance from the nearest seed
            distance_to_nearest_seed = torch.min(distances[:, :i], dim=1)[0]  # Shape: [n]
            # selected_seed_index = torch.argmax(distance_to_nearest_seed)
            selected_seed_index = torch.multinomial(distance_to_nearest_seed, 1)
            selected_indices[i] = selected_seed_index
            selected_seed = torch.index_select(X, 0, selected_seed_index)[0, :]
            seeds[i, :] = selected_seed

            # Calculate distance to this selected seed
            distances[:, i] = self.distance(X, selected_seed.unsqueeze(0))[:, 0]

        return seeds

    def mean_shift_with_seeds(self, X, Z):
        """ Run mean-shift
            @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
            @param Z: a [m x d] torch.FloatTensor of seeds to run mean shift from
        """

        Z = self.seed_hill_climbing(X, Z)

        # Connected components
        cluster_labels = self.connected_components(Z)

        return cluster_labels, Z

    @abstractmethod
    def mean_shift_smart_init(self):
        pass


class GaussianMeanShift(MeanShift):

    def __init__(self, num_seeds=100, max_iters=10, epsilon=0.05,
                 sigma=1.0, subsample_factor=1, batch_size=None):
        super().__init__(num_seeds=num_seeds,
                         max_iters=max_iters,
                         epsilon=epsilon,
                         h=sigma,
                         batch_size=batch_size)
        self.subsample_factor = subsample_factor  # Must be int
        self.distance = euclidean_distances
        self.kernel = gaussian_kernel

    def mean_shift_smart_init(self, X, sigmas=None):
        """ Run mean shift with carefully selected seeds
            @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
            @param sigmas: a [n] torch.FLoatTensor of values for per-datapoint sigmas
                           If None, use pre-specified value of sigma for all datapoints
            @return: a [n] array of cluster labels
        """
        subsampled_X = X[::self.subsample_factor, ...]  # Shape: [n//subsample_factor x d]
        if sigmas is not None:
            subsampled_sigmas = sigmas[::self.subsample_factor]  # Shape: [n//subsample_factor]
            self.h = subsampled_sigmas.unsqueeze(0)  # Shape: [1 x n//subsample_factor]

        # Get the seeds and subsampled points
        seeds = self.select_smart_seeds(subsampled_X)

        # Run mean shift
        seed_cluster_labels, updated_seeds = self.mean_shift_with_seeds(subsampled_X, seeds)

        # Get distances to updated seeds
        distances = self.distance(X, updated_seeds)

        # Get clusters by assigning point to closest seed
        closest_seed_indices = torch.argmin(distances, dim=1)  # Shape: [n]
        cluster_labels = seed_cluster_labels[closest_seed_indices]

        # Save cluster centers and labels
        uniq_labels = torch.unique(seed_cluster_labels)
        uniq_cluster_centers = torch.zeros((uniq_labels.shape[0], updated_seeds.shape[1]), dtype=torch.float,
                                           device=updated_seeds.device)
        for i, label in enumerate(uniq_labels):
            uniq_cluster_centers[i, :] = updated_seeds[seed_cluster_labels == i, :].mean(dim=0)
        self.uniq_cluster_centers = uniq_cluster_centers
        self.uniq_labels = uniq_labels

        return cluster_labels.to(X.device)  # Put it back on the device


class WeightedLoss(nn.Module):

    def __init__(self):
        super(WeightedLoss, self).__init__()
        self.weighted = False

    def generate_weight_mask(self, mask, to_ignore=None):
        """ Generates a weight mask where pixel weights are inversely proportional to
            how many pixels are in the class
            @param mask: a [N x ...] torch.FloatTensor with values in {0, 1, 2, ..., K+1}, where K is number of objects. {0,1} are background/table.
            @param to_ignore: a list of classes (integers) to ignore when creating mask
            @return: a torch.FloatTensor that is same shape as mask.
        """
        N = mask.shape[0]
        if self.weighted:

            # Compute pixel weights
            weight_mask = torch.zeros_like(mask).float() # Shape: [N x H x W]. weighted mean over pixels

            for i in range(N):
                unique_object_labels = torch.unique(mask[i])
                for obj in unique_object_labels: # e.g. [0, 1, 2, 5, 9, 10]. bg, table, 4 objects

                    if to_ignore is not None and obj in to_ignore:
                        continue

                    num_pixels = torch.sum(mask[i] == obj, dtype=torch.float)
                    weight_mask[i, mask[i] == obj] = 1 / num_pixels # inversely proportional to number of pixels

        else:
            weight_mask = torch.ones_like(mask) # mean over observed pixels
            if to_ignore is not None:
                for obj in to_ignore:
                    weight_mask[mask == obj] = 0

        return weight_mask


class CELossWeighted(WeightedLoss):
    """ Compute weighted CE loss with logits
    """

    def __init__(self, weighted=False):
        super(CELossWeighted, self).__init__()
        self.CrossEntropyLoss = nn.CrossEntropyLoss(reduction='none')
        self.weighted = weighted

    def forward(self, x, target):
        """ Compute weighted cross entropy
            @param x: a [N x C x H x W] torch.FloatTensor of values
            @param target: a [N x H x W] torch.LongTensor of values
        """
        temp = self.CrossEntropyLoss(x, target) # Shape: [N x H x W]
        weight_mask = self.generate_weight_mask(target)
        loss = torch.sum(temp * weight_mask) / torch.sum(weight_mask)

        return loss


class CELossWeightedMasked(WeightedLoss):
    """ Compute weighted CE loss with logits
    """

    def __init__(self, weighted=False):
        super(CELossWeightedMasked, self).__init__()
        self.CrossEntropyLoss = nn.CrossEntropyLoss(reduction='none')
        self.weighted = weighted

    def forward(self, x, target, fg_mask):
        """ Compute weighted cross entropy
            @param x: a [N x C x H x W] torch.FloatTensor of values
            @param target: a [N x H x W] torch.LongTensor of values
            @param fg_mask: a [N x H x W] torch.LongTensor of values in {0, 1, 2, ...}
        """

        temp = self.CrossEntropyLoss(x, target) # Shape: [N x H x W]
        weight_mask = self.generate_weight_mask(fg_mask, to_ignore=[0]) # ignore bg/table
        loss = torch.sum(temp * weight_mask) / torch.sum(weight_mask)

        return loss

def create_M_GT(foreground_labels):
    """ Create GT for Cross-Entropy Loss for M tensor.
        NOTE: no bg/table. obj index starts from 0.
        @param foreground_labels: a [H x W] torch.FloatTensor of values in {0, 1, 2, ..., K-1}
    """
    new_label = torch.zeros_like(foreground_labels)

    obj_index = 0
    for k in torch.unique(foreground_labels):

        if k in [0]:
            continue

        # Objects get labels from 0, 1, 2, ..., K-1
        new_label[foreground_labels == k] = obj_index
        obj_index += 1

    return new_label.long()

class BCEWithLogitsLossWeighted(WeightedLoss):
    """ Compute weighted BCE loss with logits
    """
    def __init__(self, weighted=False):
        super(BCEWithLogitsLossWeighted, self).__init__()
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction='none')
        self.weighted = weighted

    def forward(self, x, target):
        """ Compute masked cosine similarity loss
            @param x: a [N x H x W] torch.FloatTensor of foreground logits
            @param target: a [N x H x W] torch.FloatTensor of values in [0, 1]
        """
        temp = self.BCEWithLogitsLoss(x, target) # Shape: [N x H x W]. values are in [0, 1]
        weight_mask = self.generate_weight_mask(target)
        loss = torch.sum(temp * weight_mask) / torch.sum(weight_mask)

        return loss

class SmoothL1LossWeighted(WeightedLoss):
    """ Compute weighted Smooth L1 loss
    """
    def __init__(self, weighted=False):
        super(SmoothL1LossWeighted, self).__init__()
        self.SmoothL1Loss = nn.SmoothL1Loss(reduction='none')
        self.weighted = weighted

    def forward(self, x, target, mask=None):
        """ Compute masked cosine similarity loss
            @param x: a [N x C x H x W] torch.FloatTensor of values
            @param target: a [N x C x H x W] torch.FloatTensor of values
            @param mask: a [N x H x W] torch.FloatTensor with values in {0, 1, 2, ..., K+1}, where K is number of objects. {0,1} are background/table.
                                       Could also be None
        """
        temp = self.SmoothL1Loss(x, target).sum(dim=1) # Shape: [N x H x W]
        if mask is None:
            return torch.sum(temp) / temp.numel() # return mean

        weight_mask = self.generate_weight_mask(mask)
        loss = torch.sum(temp * weight_mask) / torch.sum(weight_mask)

        return loss

class ClusterLossWeighted(WeightedLoss):
    """ Compute Cluster loss
    """
    def __init__(self, delta, weighted=False):
        super(ClusterLossWeighted, self).__init__()
        self.weighted=weighted
        self.delta = delta

    def forward(self, x1, y1, x2, y2):
        """ Compute loss
            @param x1: a [M x D] torch.FloatTensor
            @param y1: a [M] torch.LongTensor
            @param x2: a [N x D] torch.FloatTensor
            @param y2: a [N] torch.LongTensor
            NOTE: N is NOT batch size
        """
        weight_vector_1 = self.generate_weight_mask(y1.unsqueeze(0))[0] # Shape: [M]
        weight_vector_2 = self.generate_weight_mask(y2.unsqueeze(0))[0] # Shape: [N]
        weight_matrix = torch.ger(weight_vector_1, weight_vector_2) # Shape: [M x N]
        indicator_matrix = (y1.unsqueeze(1) == y2.unsqueeze(0)).long() # Shape: [M x N]
        distance_matrix = euclidean_distances(x1,x2) # Shape: [M x N]

        positive_loss_matrix = indicator_matrix * distance_matrix**2

        negative_loss_matrix = (1 - indicator_matrix) * torch.clamp(self.delta - distance_matrix, min=0)**2

        return (weight_matrix * (positive_loss_matrix + negative_loss_matrix)).sum()