import sys
import pickle
from re import A
from cv2 import setUseOpenVX
import torch.nn as nn
import torch
import numpy as np
from keras.utils.np_utils import *
from torch.autograd import Function
inputs_dim, featre_dim = 310, 128
labels_dim, domain_dim = 64, 64
labels_num, domain_num = 3, 5

class ReverseLayerF(Function):
  @staticmethod
  def forward(ctx, x, alpha):
    ctx.alpha = alpha
    return x.view_as(x)
  @staticmethod
  def backward(ctx, grad_output):
    output = grad_output.neg() * ctx.alpha
    return output, None

class CNNModel(nn.Module):
  def __init__(self):
    super(CNNModel, self).__init__()
    self.feature = nn.Sequential()
    self.feature.add_module('jack1',nn.Linear(310,64))
    self.feature.add_module('f_relu1', nn.ReLU(True))
    self.feature.add_module('jack2', nn.Linear(64, 5))
    self.feature.add_module('f_relu2', nn.ReLU(True))

    self.class_classifier = nn.Sequential()
    self.class_classifier.add_module('c_fc1', nn.Linear(310, 100))
    # self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
    self.class_classifier.add_module('c_relu1', nn.ReLU(True))
    self.class_classifier.add_module('c_drop1', nn.Dropout())
    self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
    # self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
    self.class_classifier.add_module('c_relu2', nn.ReLU(True))
    self.class_classifier.add_module('c_fc3', nn.Linear(100, 3))
    self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

    self.domain_classifier = nn.Sequential()
    self.domain_classifier.add_module('d_fc1', nn.Linear(310, 100))
    # self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
    self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
    self.domain_classifier.add_module('d_fc2', nn.Linear(100, 5))
    self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

  def forward(self, input_data):
    # feature = self.feature(input_data)
    # feature = feature.view(-1,310)
    # print('shushuben\t',input_data.shape)
    # reverse_feature = ReverseLayerF.apply(feature, alpha)
    class_output = self.class_classifier(input_data)
    domain_output = self.domain_classifier(input_data)

    return class_output, domain_output
    
class DANN:
  def __init__(self,lr = 1e-3,batch_size = 128,image_size = 28,n_epoch = 100):
    with open('./data.pkl', 'rb') as f:
      self.data = pickle.load(f)
    self.my_net = CNNModel()
    self.optimizer = torch.optim.Adam(self.my_net.parameters(), lr=lr)
    self.loss_class = torch.nn.MSELoss()
    self.loss_domain = torch.nn.MSELoss()
    
        
  def fetch_data(self,fold):
    train_key, train_index = list(self.data.keys()), list(range(domain_num))
    valid_key, _ = train_key.pop(fold), train_index.pop(fold)
    valid_X, valid_y = self.data[valid_key]['data'], self.data[valid_key]['label']
    train_X, train_y = np.vstack([self.data[k]['data'] for k in train_key]), np.hstack([self.data[k]['label'] for k in train_key])
    valid_d, train_d = np.ones(valid_y.size) * fold, np.repeat(train_index, valid_y.size)
    valid_y  = torch.nn.functional.one_hot(torch.tensor(valid_y + 1).to(torch.int64))# jack
    valid_d = torch.nn.functional.one_hot(torch.tensor(valid_d).to(torch.int64),num_classes=5)# jack
    train_y = torch.nn.functional.one_hot(torch.tensor(train_y + 1).to(torch.int64))#to_categorical(train_y + 1).astype(np.float32)
    train_d = torch.nn.functional.one_hot(torch.tensor(train_d).to(torch.int64),num_classes=5)#to_categorical(train_d, num_classes=domain_num).astype(np.float32)
    train_X, valid_X = train_X.astype(np.float32), valid_X.astype(np.float32)
    return (torch.tensor(train_X), torch.tensor(train_y), torch.tensor(train_d)), (torch.tensor(valid_X), torch.tensor(valid_y), torch.tensor(valid_d))
  
  def train(self,n_epoch,batch_size,train,test):
    s_img, s_label,domain_label = train
    t_img,t_label,valid_d=test

    for epoch in range(n_epoch):
      index_batch = 0
      while((index_batch)*batch_size < len(s_label)):
        print('indexss\t',(index_batch+1)*batch_size)
        # p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        p = float(epoch + epoch * n_epoch) / n_epoch / n_epoch
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # training model using source data
        self.my_net.zero_grad()
        # batch_size = len(s_label)
        if ((index_batch+1)*batch_size < len(s_label)):
          s_img_mini = s_img[index_batch*batch_size:(index_batch+1)*batch_size,:]
          s_label_mini = s_label[index_batch*batch_size:(index_batch+1)*batch_size,:]
          domain_label_mini = domain_label[index_batch*batch_size:(index_batch+1)*batch_size,:]
          
        else:
          s_img_mini = s_img[index_batch*batch_size:,:]
          s_label_mini = s_label[index_batch*batch_size:,:]
          domain_label_mini = domain_label[index_batch*batch_size:,:]


        class_output, domain_output = self.my_net(input_data=s_img_mini.to(torch.float32))
        err_s_label = self.loss_class(class_output.to(torch.float32), s_label_mini.to(torch.float32))
        err_s_domain = self.loss_domain(domain_output.to(torch.float32), domain_label_mini.to(torch.float32))

        # training model using target data
        # domain_label = torch.ones(batch_size).long()

        # # if cuda:
        # #     t_img = t_img.cuda()
        # #     domain_label = domain_label.cuda()
        if ((index_batch+1)*batch_size < len(s_label)):
          t_img_mini = s_img[index_batch*batch_size:(index_batch+1)*batch_size,:]

        else:
          t_img_mini = s_img[index_batch*batch_size:,:]

        _, domain_output = self.my_net(input_data=t_img_mini.to(torch.float32))
        err_t_domain = self.loss_domain(domain_output.to(torch.float32), valid_d.to(torch.float32))
        err = err_t_domain + err_s_domain + err_s_label
        err.backward()
        self.optimizer.step()

        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
              % (epoch, epoch + 1, n_epoch, err_s_label.data.cpu().numpy(),
                 err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))
        sys.stdout.flush()
        # torch.save(self.my_net, '{0}/mnist_mnistm_model_epoch_current.pth'.format(model_root))
        index_batch += 1

       

  '''
  torch.Size([13588, 310])
  torch.Size([13588, 3])
  torch.Size([13588, 5])
  torch.Size([3397, 310])
  torch.Size([3397, 3])
  torch.Size([3397, 5])
  '''
  def run(self):
    for folder in range(domain_num):
      train, test = self.fetch_data(folder)

      # print(train[0].shape)
      # print(train[1].shape)
      # print(train[2].shape)
      # print(test[0].shape)
      # print(test[1].shape)
      # print(test[2].shape)
      # print(test[2])
      self.train(5,3397,train, test)
      # self.test_init(test)

if __name__ == '__main__':
  print('jacjajcajcjs')
  model = DANN()
  model.run()


    