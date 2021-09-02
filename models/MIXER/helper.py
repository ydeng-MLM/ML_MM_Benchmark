import numpy as np
import torch


class MyDataset(torch.utils.data.Dataset):
  def __init__(self, X, Y):
        'Initialization'
        self.X = X
        self.Y = Y

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

  def __getitem__(self, index):
        'Generates one sample of data'
        X = self.X[index]
        Y = self.Y[index]

        return X, Y

"""# Helper Functions"""

def train_test_split(X,Y,seed=42):
  np.random.seed(seed)
  indices = np.random.permutation(range(len(X)))
  train_indices = indices[:len(X)//10*7]
  val_indices = indices[len(X)//10*7:len(X)//10*8]
  test_indices = indices[len(X)//10*8:]

  X=torch.tensor(X).float()
  Y=torch.tensor(Y).float()

  return X[train_indices],Y[train_indices],X[val_indices],Y[val_indices],X[test_indices],Y[test_indices]

def train_val_split(X,Y,seed=42):
      np.random.seed(seed)
      indices = np.random.permutation(range(len(X)))
      train_indices = indices[:len(X)//10*7]
      val_indices = indices[len(X)//10*7:]

      X=torch.tensor(X).float()
      Y=torch.tensor(Y).float()

      return X[train_indices],Y[train_indices],X[val_indices],Y[val_indices]

def eval_loader(model,loader,device,criterion):
      model.eval()
      losses=[]
     
      for data in loader:
            x, y = data
            predict = model(x.to(device))
            
            loss = criterion(predict,y.to(device)).item()
            losses.append(loss)

      return np.mean(losses)

def normalize_np(x, x_max_list=None, x_min_list=None):
    """
    Normalize the x into [-1, 1] range in each dimension [:, i]
    :param x: np array to be normalized
    :return: normalized np array
    """
    if x_max_list is not None: 
        if x_min_list is None or len(x[0]) != len(x_max_list) or len(x_max_list) != len(x_min_list):
            print("In normalize_np, your dimension does not match with provided x_max, try again")
            quit()

    new_x_max_list = []
    new_x_min_list = []
    for i in range(len(x[0])):
        if x_max_list is None:
            x_max = np.max(x[:, i])
            x_min = np.min(x[:, i])
        else:
            x_max = x_max_list[i]
            x_min = x_min_list[i]
        x_range = (x_max - x_min ) /2.
        x_avg = (x_max + x_min) / 2.
        x[:, i] = (x[:, i] - x_avg) / x_range
        print("In normalize_np, row ", str(i), " your max is:", np.max(x[:, i]))
        print("In normalize_np, row ", str(i), " your min is:", np.min(x[:, i]))
        if x_max_list is None:
            assert np.max(x[:, i]) - 1 < 0.0001, 'your normalization is wrong'
            assert np.min(x[:, i]) + 1 < 0.0001, 'your normalization is wrong'
            new_x_max_list.append(x_max)
            new_x_min_list.append(x_min)
    return x, np.array(new_x_max_list), np.array(new_x_min_list)