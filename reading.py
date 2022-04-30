import sys
import pickle
from re import A
from cv2 import setUseOpenVX
import torch.nn as nn
import torch
import numpy as np
# from keras.utils.np_utils import *
from torch.autograd import Function
torch.cuda.set_device(1)
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
    # self.bias_class = torch.autograd.VariableVariable(torch.FloatTensor([128]),requires_grad=True)
    # self.bias_domain = torch.autograd.VariableVariable(torch.FloatTensor([]),requires_grad=True)
    super(CNNModel, self).__init__()
    self.feature = nn.Sequential()
    self.feature.add_module('c_fc1', nn.Linear(310, 128))
    self.feature.add_module('c_bn1', nn.BatchNorm1d(128))
    self.feature.add_module('c_relu1', nn.ReLU(True))
    self.feature.add_module('c_drop1', nn.Dropout())
    self.feature.add_module('c_fc2', nn.Linear(128, 100))
    self.feature.add_module('c_bn2', nn.BatchNorm1d(100))
    self.feature.add_module('c_relu2', nn.ReLU(True))
    self.feature.add_module('c_fc3', nn.Linear(100, 310))
    self.feature.add_module('c_softmax', nn.LogSoftmax(dim=1))
    
    self.class_classifier = nn.Sequential()
    self.class_classifier.add_module('c_fc1', nn.Linear(310, 128))
    self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(128))
    self.class_classifier.add_module('c_relu1', nn.ReLU(True))
    self.class_classifier.add_module('c_drop1', nn.Dropout())
    self.class_classifier.add_module('c_fc2', nn.Linear(128, 100))
    self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
    self.class_classifier.add_module('c_relu2', nn.ReLU(True))
    self.class_classifier.add_module('c_fc3', nn.Linear(100, 3))
    self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

    self.domain_classifier = nn.Sequential()
    self.domain_classifier.add_module('d_fc1', nn.Linear(310, 128))
    self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(128))
    self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
    self.domain_classifier.add_module('d_fc2', nn.Linear(128, 5))
    self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

  def forward(self, input_data,alpha):
    feature = self.feature(input_data)
    feature = feature.view(-1,310)
    # print('shushuben\t',input_data.shape)
    reverse_feature = ReverseLayerF.apply(feature, alpha)
    class_output = self.class_classifier(feature)
    
    domain_output = self.domain_classifier(reverse_feature)

    return class_output, domain_output
    
class DANN:
  def __init__(self,lr = 1e-3,batch_size = 128,image_size = 28,n_epoch = 100):
    with open('./data.pkl', 'rb') as f:
      self.data = pickle.load(f)
    self.my_net = CNNModel().cuda()
    self.optimizer = torch.optim.SGD(self.my_net.parameters(), lr=lr)
    self.loss_class = torch.nn.MSELoss().cuda()
    self.loss_domain = torch.nn.MSELoss().cuda()

  def acc_rate(self,predict,labels):
    counter = 0
    zeross = torch.zeros(1,predict.shape[1]).cuda()
    for i,j in zip(predict,labels):
      temp = torch.zeros(1,labels.shape[1]).cuda()
      temp[:,int(i.argmax(dim=0))] = 1
      # print(temp*j)
      if zeross.equal(temp*(j.cuda())):
        pass
      else:
        counter += 1
    return counter/predict.shape[0]
        
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
  
  def test(self,test):
    # to do
    t_img,t_label,valid_d=test
    t_img ,t_label,valid_d = t_img.cuda() ,t_label.cuda(),valid_d.cuda()
    p = float(len(t_label)*len(t_label))
    alpha = 2. / (1. + np.exp(-10 * p)) - 1
    class_output, valid_output = self.my_net(t_img.to(torch.float32), alpha)
    acc_test = self.acc_rate(class_output,t_label)
    acc_domain = self.acc_rate(valid_output,valid_d)
    print('\ntest acc is {} the acc_domain is {}\n'.format(acc_test,acc_domain))

  def train(self,n_epoch,batch_size,train,test,folder):
    s_img, s_label,domain_label = train
    # print(domain_label.shape,'\n',s_label.shape),s_img.shape,s_label.shape)
    # sys.exit()
    t_img,t_label,domain_t = test
    for epoch in range(n_epoch):
      index_batch = 0
      while((index_batch)*batch_size < len(s_label)):
        p = float(index_batch + epoch * len(s_label)) / n_epoch /len(s_label)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        # training model using source data
        self.my_net.zero_grad()

        if ((index_batch+1)*batch_size < len(s_label)):
          s_img_mini = s_img[index_batch*batch_size:(index_batch+1)*batch_size,:].cuda()
          s_label_mini = s_label[index_batch*batch_size:(index_batch+1)*batch_size,:].cuda()
          domain_label_mini =domain_label[index_batch*batch_size:(index_batch+1)*batch_size,:].cuda()
          # print('\n\nid_num+index_batch*batch_size: \n\n',id_num+index_batch*batch_size,'\n',id_num+(index_batch+1)*batch_size)
        elif((index_batch)*batch_size <= len(s_label)):
          s_img_mini = s_img[index_batch*batch_size:,:].cuda()
          s_label_mini = s_label[index_batch*batch_size:,:].cuda()
          domain_label_mini =domain_label[index_batch*batch_size:,:].cuda()
          # print('\n\nid_num+index_batch*batch_size: \n\n',id_num+index_batch*batch_size,'\n',id_num+index_batch*batch_size+len(s_label_mini))
        # print(s_img_mini.shape)
        class_output, domain_output = self.my_net(s_img_mini.to(torch.float32),alpha)
        # print('classes\t',class_output.shape)
        # print('domain_output\t',class_output.shape,s_label_mini.shape)
        # sys.exit()
        # print(domain_label_mini.shape,s_label_mini.shape,s_img_mini.shape)
        err_s_label = self.loss_class(class_output.to(torch.float32), s_label_mini.to(torch.float32))
        err_s_domain = self.loss_domain(domain_output.to(torch.float32), domain_label_mini.to(torch.float32))

        if ((index_batch+1)*batch_size < len(s_label)):
          t_img_mini = t_img[int(index_batch*batch_size/4):int((index_batch+1)*batch_size/4),:].cuda()
          domain_t_label_mini = domain_t[int(index_batch*batch_size/4):int((index_batch+1)*batch_size/4),:].cuda()

        elif((index_batch)*batch_size <= len(s_label)):
          t_img_mini = t_img[int(index_batch*batch_size/4):,:].cuda()
          domain_t_label_mini = domain_t[int(index_batch*batch_size/4):,:].cuda()
        
        # print('t_img_mini\t',t_img_mini.shape)

        _, domain_output = self.my_net(t_img_mini.to(torch.float32),alpha)
        # print('domain_out\t',domain_output.shape,domain_t_label_mini.shape)
        # sys.exit()
        err_t_domain = self.loss_domain(domain_output.to(torch.float32), domain_t_label_mini.to(torch.float32))
        err = err_t_domain + err_s_domain + err_s_label
        err.backward()
        self.optimizer.step()

        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
              % (epoch, epoch + 1, n_epoch, err_s_label.data.cpu().numpy(),
                 err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))
        sys.stdout.flush()
        # torch.save(self.my_net, '{0}/mnist_mnistm_model_epoch_current.pth'.format(model_root))
        index_batch += 1
      # print("\n arrurate rate is %.4f"%(self.acc_rate(class_output,s_label_mini)))

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
      self.train(10,400,train,test, folder)
      self.test(test)
      # self.test_init(test)

if __name__ == '__main__':
  model = DANN()
  model.run()


    