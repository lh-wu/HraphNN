
from scipy.spatial.distance import pdist,squareform
from dataloader import get_data
import numpy as np
from collections import Counter
from sklearn.metrics import euclidean_distances
from sklearn.base import TransformerMixin
from metric_learn._util import _initialize_components, _check_n_components
from metric_learn.base_metric import MahalanobisMixin


class LMNN(MahalanobisMixin, TransformerMixin):
  def __init__(self, init='auto', k=3, min_iter=50, max_iter=1000,
               learn_rate=1e-7, regularization=0.5, convergence_tol=0.001,
               verbose=False, preprocessor=None,
               n_components=None, random_state=None):
    self.init = init
    self.k = k
    self.min_iter = min_iter
    self.max_iter = max_iter
    self.learn_rate = learn_rate
    self.regularization = regularization
    self.convergence_tol = convergence_tol
    self.verbose = verbose
    self.n_components = n_components
    self.random_state = random_state
    super(LMNN, self).__init__(preprocessor)

  def fit(self, X, y,hanming_matrix):
    k = self.k
    reg = self.regularization
    learn_rate = self.learn_rate
    num_pts, d = X.shape
    output_dim = _check_n_components(d, self.n_components)
    _, label_inds = np.unique(y, return_inverse=True)
    if len(label_inds) != num_pts:
      raise ValueError('Must have one label per point.')

    self.components_ = _initialize_components(output_dim, X, y, self.init,
                                              self.verbose,
                                              random_state=self.random_state)
    # required_k = np.bincount(label_inds).min()
    # if self.k > required_k:
    #   raise ValueError('not enough class labels for specified k'
    #                    ' (smallest class has %d)' % required_k)

    target_neighbors, target_neighbors_hanming = self._select_targets(X, hanming_matrix, k)

    # sum outer products
    dfG = _sum_outer_products(X, target_neighbors.flatten(),
                              np.repeat(np.arange(X.shape[0]), k))

    # initialize L
    L = self.components_

    # first iteration: we compute variables (including objective and gradient)
    #  at initialization point
    G, objective, total_active = self._loss_grad(X, L, dfG, k,
                                                 reg, target_neighbors,target_neighbors_hanming,hanming_matrix)

    it = 1  # we already made one iteration

    if self.verbose:
      print("iter | objective | objective difference | active constraints",
            "| learning rate")

    # main loop
    for it in range(2, self.max_iter):
      # then at each iteration, we try to find a value of L that has better
      # objective than the previous L, following the gradient:
      while True:
        # the next point next_L to try out is found by a gradient step
        L_next = L - learn_rate * G
        # we compute the objective at next point
        # we copy variables that can be modified by _loss_grad, because if we
        # retry we don t want to modify them several times
        (G_next, objective_next, total_active_next) = (
            self._loss_grad(X, L_next, dfG, k, reg, target_neighbors,target_neighbors_hanming,hanming_matrix))
        assert not np.isnan(objective)
        delta_obj = objective_next - objective
        if delta_obj > 0:
          # if we did not find a better objective, we retry with an L closer to
          # the starting point, by decreasing the learning rate (making the
          # gradient step smaller)
          learn_rate /= 2
        else:
          # otherwise, if we indeed found a better obj, we get out of the loop
          break
      # when the better L is found (and the related variables), we set the
      # old variables to these new ones before next iteration and we
      # slightly increase the learning rate
      L = L_next
      G, objective, total_active = G_next, objective_next, total_active_next
      learn_rate *= 1.01

      if self.verbose:
        print(it, objective, delta_obj, total_active, learn_rate)

      # check for convergence
      if it > self.min_iter and abs(delta_obj) < self.convergence_tol:
        if self.verbose:
          print("LMNN converged with objective", objective)
        break
    else:
      if self.verbose:
        print("LMNN didn't converge in %d steps." % self.max_iter)

    # store the last L
    self.components_ = L
    self.n_iter_ = it
    return self,L

  def _loss_grad(self, X, L, dfG, k, reg, target_neighbors,target_neighbors_hanming, hanming_matrix):
    # Compute pairwise distances under current metric
    Lx = L.dot(X.T).T

    # we need to find the furthest neighbor:
    Ni = 1 + _inplace_paired_L2(Lx[target_neighbors], Lx[:, None, :])
    furthest_neighbors = self._select_furthest_neighbor(Ni,target_neighbors,target_neighbors_hanming)
    impostors,impostors_hanming_list = self._find_impostors(furthest_neighbors, X, L, hanming_matrix)

    g0 = _inplace_paired_L2(*Lx[impostors])

    # we reorder the target neighbors
    g1, g2 = Ni[impostors]
    g1_hm,g2_hm=target_neighbors_hanming[impostors]
    # compute the gradient
    total_active = 0
    df = np.zeros((X.shape[1], X.shape[1]))
    for nn_idx in reversed(range(k)):  # note: reverse not useful here
      # act1 = g0 < g1[:, nn_idx] & impostors_hanming_list<g1_hm[:,nn_idx]
      # act2 = g0 < g2[:, nn_idx] & impostors_hanming_list<g2_hm[:,nn_idx]

      act1_1 = g0 < g1[:, nn_idx]
      act1_2 = impostors_hanming_list < g1_hm[:, nn_idx]
      act1 = act1_1 & act1_2
      act2_1 = g0 < g2[:, nn_idx]
      act2_2 = impostors_hanming_list < g2_hm[:, nn_idx]
      act2 = act2_1 & act2_2


      total_active += act1.sum() + act2.sum()

      targets = target_neighbors[:, nn_idx]
      PLUS, pweight = _count_edges(act1, act2, impostors, targets)
      df += _sum_outer_products(X, PLUS[:, 0], PLUS[:, 1], pweight)

      in_imp, out_imp = impostors
      df -= _sum_outer_products(X, in_imp[act1], out_imp[act1])
      df -= _sum_outer_products(X, in_imp[act2], out_imp[act2])

    # do the gradient update
    assert not np.isnan(df).any()
    G = dfG * reg + df * (1 - reg)
    G = L.dot(G)
    # compute the objective function
    objective = total_active * (1 - reg)
    objective += G.flatten().dot(L.flatten())
    return 2 * G, objective, total_active

  def _select_targets(self,X, hanming_matrix,k):
      subj_num = X.shape[0]
      target_neighbors = np.empty((subj_num, k), dtype=np.int32)
      target_neighbors_hanming = np.empty((subj_num, k), dtype=np.int32)
      for i in range(subj_num):
          temp = hanming_matrix[i, :]
          neighbor_inds = np.argsort(temp)[::-1][0:k]
          target_neighbors[i, :] = neighbor_inds
          target_neighbors_hanming[i,:] = temp[neighbor_inds]
      return target_neighbors,target_neighbors_hanming

  def _select_furthest_neighbor(self,Ni, target_neighbors, target_neighbors_hanming):
      result = []
      for i in range(target_neighbors.shape[0]):
          i_furthest_neighbor = []
          i_neighbor_inds = target_neighbors[i, :]
          priority_list = target_neighbors_hanming[i, :]
          dist_relevant_inds = np.argsort(Ni[i, :])[::-1]
          _sel_bin = []
          for k in dist_relevant_inds:
              if priority_list[k] in _sel_bin:
                  pass
              else:
                  i_furthest_neighbor.append((i, i_neighbor_inds[k], Ni[i, :][k], priority_list[k]))
                  _sel_bin.append(priority_list[k])
          result.append(i_furthest_neighbor)
      return result

  def _find_impostors(self, furthest_neighbors, X, L, hanming_matrix):
    Lx = X.dot(L.T)
    all_distance=squareform(pdist(Lx))
    result=[]
    hanming_list=[]
    for i in range(X.shape[0]):
        i_fneighbor_list=furthest_neighbors[i]
        for (src,tar,margin,hm_dist) in i_fneighbor_list:
            for j in range(X.shape[0]):
                if hanming_matrix[i][j]<hm_dist and all_distance[i][j]<margin:
                    result.append([i,j])
                    hanming_list.append(hanming_matrix[i][j])
    result=np.array(result)
    hanming_list=np.array(hanming_list)
    return result.T,hanming_list


def _inplace_paired_L2(A, B):
  '''Equivalent to ((A-B)**2).sum(axis=-1), but modifies A in place.'''
  A -= B
  return np.einsum('...ij,...ij->...i', A, A)


def _count_edges(act1, act2, impostors, targets):
  imp = impostors[0, act1]
  c = Counter(zip(imp, targets[imp]))
  imp = impostors[1, act2]
  c.update(zip(imp, targets[imp]))
  if c:
    active_pairs = np.array(list(c.keys()))
  else:
    active_pairs = np.empty((0, 2), dtype=int)
  return active_pairs, np.array(list(c.values()))


def _sum_outer_products(data, a_inds, b_inds, weights=None):
  Xab = data[a_inds] - data[b_inds]
  if weights is not None:
    return np.dot(Xab.T, Xab * weights[:, None])
  return np.dot(Xab.T, Xab)


#######################################
def get_hanming_matrix(ph_data):
    # phonetic_data——0age 1gender 2edu 3marry 4home 5eth 6race
    subj_num,fea_num=ph_data.shape
    hanming_matrix=np.zeros((subj_num,subj_num),dtype=np.int32)
    for i in range(fea_num):
        temp=ph_data[:,i]
        if i==0:
            for j in range(subj_num):
                for k in range(j+1,subj_num):
                    if np.abs(temp[j]-temp[k])<=2.0:
                        hanming_matrix[j][k]=hanming_matrix[j][k]+1
                        hanming_matrix[k][j] = hanming_matrix[k][j] + 1
        else:
            for j in range(subj_num):
                for k in range(j+1,subj_num):
                    if temp[j]==temp[k]:
                        hanming_matrix[j][k]=hanming_matrix[j][k]+1
                        hanming_matrix[k][j] = hanming_matrix[k][j] + 1
    return hanming_matrix


def _select_targets(X, hanming_matrix,k=8):
    subj_num=X.shape[0]
    target_neighbors = np.empty((subj_num, k), dtype=np.int32)
    for i in range(subj_num):
        temp=hanming_matrix[i,:]
        neighbor_inds=np.argsort(temp)[::-1][0:k]
        target_neighbors[i,:]=neighbor_inds
    return target_neighbors

def _select_furthest_neighbor(Ni,target_neighbors,hanming_matirx):
    result=[]
    for i in range(target_neighbors.shape[0]):
        i_furthest_neighbor=[]
        i_neighbor_inds=target_neighbors[i,:]
        priority_list=hanming_matirx[i,:][i_neighbor_inds]
        dist_relevant_inds=np.argsort(Ni[i,:])[::-1]
        _sel_bin=[]
        for k in dist_relevant_inds:
            if priority_list[k] in _sel_bin:
                pass
            else:
                i_furthest_neighbor.append((i,i_neighbor_inds[k],Ni[i,:][k],priority_list[k]))
                _sel_bin.append(priority_list[k])
        result.append(i_furthest_neighbor)
    return result