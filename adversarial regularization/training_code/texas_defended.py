
# coding: utf-8

# In[1]:


from __future__ import print_function


import argparse
import pickle
import os
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='7' 
import shutil
import time
import random
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
import numpy as np


# In[2]:


from sklearn.cluster import KMeans
from sklearn import datasets

import urllib
import tarfile
# In[3]:


use_cuda = torch.cuda.is_available()


# In[20]:


DATASET_PATH = '../datasets/texas/'


if not os.path.isdir(DATASET_PATH):
    mkdir_p(DATASET_PATH)

DATASET_FEATURES = os.path.join(DATASET_PATH,'texas/100/feats')
DATASET_LABELS = os.path.join(DATASET_PATH,'texas/100/labels')


if not os.path.isfile(DATASET_FEATURES):
    print("Dowloading the dataset...")
    urllib.request.urlretrieve("https://www.comp.nus.edu.sg/~reza/files/dataset_texas.tgz",os.path.join(DATASET_PATH,'tmp.tgz'))
    print('Dataset Dowloaded')

    tar = tarfile.open(os.path.join(DATASET_PATH,'tmp.tgz'))
    tar.extractall(path=DATASET_PATH)


# In[21]:


data_set_features =np.genfromtxt(DATASET_FEATURES,delimiter=',')
data_set_label =np.genfromtxt(DATASET_LABELS,delimiter=',')


# In[22]:


X =data_set_features.astype(np.float64)
Y = data_set_label.astype(np.int32)-1


# In[23]:


class Texas(nn.Module):
    def __init__(self,num_classes=100):
        super(Texas, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(6169,1024),
            nn.Tanh(),
            nn.Linear(1024,512),
            nn.Tanh(),
            nn.Linear(512,256),
            nn.Tanh(),
            nn.Linear(256,128),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(128,num_classes)
#         for key in self.state_dict():
#             if key.split('.')[-1] == 'weight':    
#                 nn.init.normal(self.state_dict()[key], std=0.01)
#                 print (key)
                
#             elif key.split('.')[-1] == 'bias':
#                 self.state_dict()[key][...] = 0
        
    def forward(self,x):
        hidden_out = self.features(x)
        
        
        
        return self.classifier(hidden_out),hidden_out


# In[24]:


class InferenceAttack_HZ(nn.Module):
    def __init__(self,num_classes):
        self.num_classes=num_classes
        super(InferenceAttack_HZ, self).__init__()
        self.features=nn.Sequential(
            nn.Linear(num_classes,512),
            nn.ReLU(),
            nn.Linear(512,64),
            nn.ReLU(),
            )
        
  
        
        self.labels=nn.Sequential(
           nn.Linear(num_classes,64),
            )
        self.combine=nn.Sequential(
            nn.Linear(64*2,64),
            nn.ReLU(),
            nn.Linear(64,1),
            )
        for key in self.state_dict():
            print (key)
            if key.split('.')[-1] == 'weight':    
                nn.init.normal(self.state_dict()[key], std=0.01)
                print (key)
                
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0
        self.output= nn.Sigmoid()
    def forward(self,x1,x2,l):
        #print (l.size(),x.size())
        out_x1 = self.features(x1)
        
        out_l = self.labels(l)
        
        
        
            
        is_member =self.combine( torch.cat((out_x1,out_l),1))
        
        
        return self.output(is_member)


# In[25]:


len_train =len(X)


r = np.load('../dataset_shuffle/random_r_texas100.npy')
X=X[r]
Y=Y[r]

train_classifier_ratio, train_attack_ratio = float(10000)/float(X.shape[0]),0.3
train_classifier_data = X[:int(train_classifier_ratio*len_train)]
train_attack_data = X[int(train_classifier_ratio*len_train):int((train_classifier_ratio+train_attack_ratio)*len_train)]
test_data = X[int((train_classifier_ratio+train_attack_ratio)*len_train):]

train_classifier_label = Y[:int(train_classifier_ratio*len_train)]
train_attack_label = Y[int(train_classifier_ratio*len_train):int((train_classifier_ratio+train_attack_ratio)*len_train)]
test_label = Y[int((train_classifier_ratio+train_attack_ratio)*len_train):]





def train(train_data,labels, model, criterion, optimizer, epoch, use_cuda,num_batchs=999999):
    # switch to train mode
    model.train()

    

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    
    end = time.time()
    len_t =  (len(train_data)//batch_size)-1
    
    for ind in range(len_t):
        if ind > num_batchs:
            break
        # measure data loading time
        inputs = train_data[ind*batch_size:(ind+1)*batch_size]
        targets = labels[ind*batch_size:(ind+1)*batch_size]

        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs,_ = model(inputs)
        
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data, inputs.size()[0])
        top1.update(prec1, inputs.size()[0])
        top5.update(prec5, inputs.size()[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if ind%100==0:
            print  ('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=ind + 1,
                    size=len_t,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    ))

    return (losses.avg, top1.avg)
        


def test(test_data,labels, model, criterion, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    len_t =  (len(test_data)//batch_size)-1
    
    for ind in range(len_t):
        # measure data loading time
        inputs = test_data[ind*batch_size:(ind+1)*batch_size]
        targets = labels[ind*batch_size:(ind+1)*batch_size]

        
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs,_ = model(inputs)
        
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data, inputs.size()[0])
        top1.update(prec1, inputs.size()[0])
        top5.update(prec5, inputs.size()[0])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
#         if ind % 100==0:
            
#             print ('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
#                         batch=ind + 1,
#                         size=len(test_data),
#                         data=data_time.avg,
#                         bt=batch_time.avg,
#                         loss=losses.avg,
#                         top1=top1.avg,
#                         top5=top5.avg,
#                         ))

    return (losses.avg, top1.avg)


# In[30]:



def train_privatly(train_data,labels, model,inference_model, criterion, optimizer, epoch, use_cuda,num_batchs=10000,skip_batch=0,alpha=0.5):
    # switch to train mode
    model.train()
    inference_model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    
    
    len_t =  (len(train_data)//batch_size)-1
    
    for ind in range(skip_batch,len_t):
        
        if ind >= skip_batch+num_batchs:
            break
        
        # measure data loading time
        
        #print (ind)
        
        inputs = train_data[ind*batch_size:(ind+1)*batch_size]
        
        
        targets = labels[ind*batch_size:(ind+1)*batch_size]

        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs,h_layer = model(inputs)
        
        
        
        one_hot_tr = torch.from_numpy((np.zeros((outputs.size()[0],outputs.size(1))))).cuda().type(torch.cuda.FloatTensor)
        target_one_hot_tr = one_hot_tr.scatter_(1, targets.type(torch.cuda.LongTensor).view([-1,1]).data,1)
        
        infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)
        
        
        
        inference_output = inference_model ( outputs,h_layer,infer_input_one_hot)
        #print (inference_output.mean())
        
        relu = nn.ReLU()
        loss = criterion(outputs, targets) + ((alpha)*(torch.mean((inference_output)) - 0.5))

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data, inputs.size()[0])
        top1.update(prec1, inputs.size()[0])
        top5.update(prec5, inputs.size()[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if ind%100==0:
            print  (alpha, '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=ind + 1,
                    size=len_t,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    ))

    return (losses.avg, top1.avg)


# In[31]:


def outputres(test_data,labels, model, criterion, epoch, use_cuda):

    

    # switch to evaluate mode
    model.eval()

    end = time.time()
    len_t =  (len(test_data)//batch_size)-1
    alloutputs = np.zeros((batch_size*(len_t),100))
    for ind in range(len_t):
        # measure data loading time
        inputs = test_data[ind*batch_size:(ind+1)*batch_size]
        targets = labels[ind*batch_size:(ind+1)*batch_size]

        
        

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs,_ = model(inputs)
        
        alloutputs[ind*batch_size:(ind+1)*batch_size,:]=F.softmax(outputs,dim=1).data.cpu().numpy()

    return alloutputs


# In[32]:

REGULARIZATION_RATIO = 10.0
def save_checkpoint(state, is_best, checkpoint='./models/texas_defended', filename='checkpoint.pth.tar'):
    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)

    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


# In[33]:


best_acc = 0.0


# In[34]:


epochs=100


# In[38]:



def train_attack(train_data,labels,attack_data,attack_label, model,attack_model, criterion,attack_criterion, optimizer,attack_optimizer, epoch, use_cuda,num_batchs=100000,skip_batch=0):
    # switch to train mode
    model.eval()
    attack_model.train()

    

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    
    end = time.time()
    len_t =  min((len(attack_data)//batch_size) ,(len(train_data)//batch_size))-1
    
    #print (skip_batch, len_t)
    
    for ind in range(skip_batch, len_t):
        
        if ind >= skip_batch+num_batchs:
            break
        # measure data loading time
        inputs = train_data[ind*batch_size:(ind+1)*batch_size]
        targets = labels[ind*batch_size:(ind+1)*batch_size]
        
        inputs_attack = attack_data[ind*batch_size:(ind+1)*batch_size]
        targets_attack = attack_label[ind*batch_size:(ind+1)*batch_size]
        #print ( len(targets_attack), len(targets))
        

        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
            inputs_attack , targets_attack = inputs_attack.cuda(), targets_attack.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        inputs_attack , targets_attack = torch.autograd.Variable(inputs_attack), torch.autograd.Variable(targets_attack)


        # compute output
        outputs, h_layer = model(inputs)
        outputs_non, h_layer_non = model(inputs_attack)
        
        classifier_input = torch.cat((inputs,inputs_attack))

        comb_inputs_h = torch.cat((h_layer,h_layer_non))
        comb_inputs = torch.cat((outputs,outputs_non))
        
        if use_cuda:
            comb_targets= torch.cat((targets,targets_attack)).view([-1,1]).type(torch.cuda.FloatTensor)
        else:
            comb_targets= torch.cat((targets,targets_attack)).view([-1,1]).type(torch.FloatTensor)
            
        #print (comb_inputs.size(),comb_targets.size())
        attack_input = comb_inputs #torch.cat((comb_inputs,comb_targets),1)
        
        
        one_hot_tr = torch.from_numpy((np.zeros((attack_input.size()[0],outputs.size(1))))).cuda().type(torch.cuda.FloatTensor)
        target_one_hot_tr = one_hot_tr.scatter_(1, torch.cat((targets,targets_attack)).type(torch.cuda.LongTensor).view([-1,1]).data,1)
        
        infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)
         
        
        
#         sf= nn.Softmax(dim=0)
        
#         att_inp=torch.stack([attack_input, infer_input_one_hot],1)
        
        
#         att_inp = att_inp.view([attack_input.size()[0],1,2,attack_input.size(1)])
        
        #attack_output = attack_model(att_inp).view([-1])
        attack_output = attack_model(attack_input,comb_inputs_h,infer_input_one_hot).view([-1])
        #attack_output = attack_model(attack_input).view([-1])
        att_labels = np.zeros((inputs.size()[0]+inputs_attack.size()[0]))
        att_labels [:inputs.size()[0]] =1.0
        att_labels [inputs.size()[0]:] =0.0
        is_member_labels = torch.from_numpy(att_labels).type(torch.FloatTensor)
        
        if use_cuda:
            is_member_labels = is_member_labels.cuda()
        
        v_is_member_labels = torch.autograd.Variable(is_member_labels)
        
        
        classifier_targets = comb_targets.clone().view([-1]).type(torch.cuda.LongTensor)

        
        loss_attack = attack_criterion(attack_output, v_is_member_labels)
        

        # measure accuracy and record loss
        #prec1,p5 = accuracy(attack_output.data, v_is_member_labels.data, topk=(1,2))
        
        prec1=np.mean(np.equal((attack_output.data.cpu().numpy() >0.5),(v_is_member_labels.data.cpu().numpy()> 0.5)))
        losses.update(loss_attack.data, attack_input.size()[0])
        top1.update(prec1, attack_input.size()[0])
        
        #print ( attack_output.data.cpu().numpy(),v_is_member_labels.data.cpu().numpy() ,attack_input.data.cpu().numpy())
        #raise
        
        
        
        # compute gradient and do SGD step
        attack_optimizer.zero_grad()
        loss_attack.backward()
        attack_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if ind%100==0:
            print  ('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} '.format(
                    batch=ind + 1,
                    size=len_t,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    ))

    return (losses.avg, top1.avg)


# In[39]:


def test_attack(train_data,labels,attack_data,attack_label, model,attack_model, criterion,attack_criterion, optimizer,attack_optimizer, epoch, use_cuda):

    model.eval()
    attack_model.eval()

    

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    
    end = time.time()
    len_t =  min((len(attack_data)//batch_size) ,(len(train_data)//batch_size))-1
    
    for ind in range(len_t):
        # measure data loading time
        inputs = train_data[ind*batch_size:(ind+1)*batch_size]
        targets = labels[ind*batch_size:(ind+1)*batch_size]
        
        inputs_attack = attack_data[ind*batch_size:(ind+1)*batch_size]
        targets_attack = attack_label[ind*batch_size:(ind+1)*batch_size]
        #print ( len(targets_attack), len(targets))
        

        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
            inputs_attack , targets_attack = inputs_attack.cuda(), targets_attack.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        inputs_attack , targets_attack = torch.autograd.Variable(inputs_attack), torch.autograd.Variable(targets_attack)


        # compute output
        outputs,h_layer = model(inputs)
        outputs_non,h_layer_non = model(inputs_attack)
        

        comb_inputs_h = torch.cat((h_layer,h_layer_non))
        comb_inputs = torch.cat((outputs,outputs_non))
        
        if use_cuda:
            comb_targets= torch.cat((targets,targets_attack)).view([-1,1]).type(torch.cuda.FloatTensor)
        else:
            comb_targets= torch.cat((targets,targets_attack)).view([-1,1]).type(torch.FloatTensor)
            
        #print (comb_inputs.size(),comb_targets.size())
        attack_input = comb_inputs #torch.cat((comb_inputs,comb_targets),1)
        
        
        one_hot_tr = torch.from_numpy((np.zeros((attack_input.size()[0],outputs.size(1))))).cuda().type(torch.cuda.FloatTensor)
        target_one_hot_tr = one_hot_tr.scatter_(1, torch.cat((targets,targets_attack)).type(torch.cuda.LongTensor).view([-1,1]).data,1)
        
        infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)
         
        

        
        #attack_output = attack_model(att_inp).view([-1])
        attack_output = attack_model(attack_input,comb_inputs_h,infer_input_one_hot).view([-1])
        #attack_output = attack_model(attack_input).view([-1])
        att_labels = np.zeros((inputs.size()[0]+inputs_attack.size()[0]))
        att_labels [:inputs.size()[0]] =1.0
        att_labels [inputs.size()[0]:] =0.0
        is_member_labels = torch.from_numpy(att_labels).type(torch.FloatTensor)
        if use_cuda:
            is_member_labels = is_member_labels.cuda()
        
        v_is_member_labels = torch.autograd.Variable(is_member_labels)
        
        
        
        
        
        
        
        
        loss = attack_criterion(attack_output, v_is_member_labels)
        

        # measure accuracy and record loss
        #prec1,p5 = accuracy(attack_output.data, v_is_member_labels.data, topk=(1,2))
        
        prec1=np.mean(np.equal((attack_output.data.cpu().numpy() >0.5),(v_is_member_labels.data.cpu().numpy()> 0.5)))
        losses.update(loss.data, attack_input.size()[0])
        top1.update(prec1, attack_input.size()[0])
        
        #raise
        
        # compute gradient and do SGD step


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if ind%100==0:
            print  ('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} '.format(
                    batch=ind + 1,
                    size=len_t,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    ))
            

    return (losses.avg, top1.avg)
        
    return 


# In[42]:


attack_model = InferenceAttack_HZ(100)
attack_model = torch.nn.DataParallel(attack_model).cuda()
attack_criterion = nn.MSELoss()
attack_optimizer = optim.Adam(attack_model.parameters(),lr=0.001)
net = Texas()
net = torch.nn.DataParallel(net).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)




# In[43]:


batch_size=128
best_acc = 0.0
for epoch in range(0, 20):
    
    r= np.arange(len(train_classifier_data))
    np.random.shuffle(r)
    train_classifier_data = train_classifier_data[r]
    train_classifier_label = train_classifier_label[r]
    
    train_classifier_data_tensor = torch.from_numpy(train_classifier_data).type(torch.FloatTensor)
    train_classifier_label_tensor = torch.from_numpy(train_classifier_label).type(torch.LongTensor)
    
    r= np.arange(len(train_attack_data))
    np.random.shuffle(r)
    
    train_attack_data = train_attack_data[r]
    train_attack_label = train_attack_label[r]
    
    train_attack_data_tensor = torch.from_numpy(train_attack_data).type(torch.FloatTensor)
    train_attack_label_tensor = torch.from_numpy(train_attack_label).type(torch.LongTensor)
    att_accs =[]
    
    
    test_data_tensor = torch.from_numpy(test_data).type(torch.FloatTensor)
    test_label_tensor = torch.from_numpy(test_label).type(torch.LongTensor)
    print('\nEpoch: [%d | %d]' % (epoch + 1, epochs))
    
    
    if epoch <1: 
        
        
        train_loss, train_acc = train(train_classifier_data_tensor,train_classifier_label_tensor, net, criterion, optimizer, epoch, use_cuda)
        for i in range(5):
            train_attack(train_classifier_data_tensor,train_classifier_label_tensor
                                             ,train_attack_data_tensor,train_attack_label_tensor,net,attack_model,criterion,attack_criterion,optimizer,attack_optimizer,epoch,use_cuda)
        
    else:
        for i in range(76):
            at_loss,at_acc = train_attack(train_classifier_data_tensor,train_classifier_label_tensor
                                                 ,train_attack_data_tensor,train_attack_label_tensor,net,attack_model,criterion,attack_criterion,optimizer,attack_optimizer,epoch,use_cuda,50,(i*50)%150)
            att_accs.append(at_acc)
            att_epoch_acc = np.mean(att_accs)
            train_privatly(train_classifier_data_tensor,train_classifier_label_tensor, net,attack_model, criterion, optimizer, epoch, use_cuda,2,(2*i)%152,REGULARIZATION_RATIO)

    
        test_loss, test_acc = test(test_data_tensor,test_label_tensor, net, criterion, epoch, use_cuda)
        #privacy_loss, privacy_acc = privacy_train(trainloader,testloader,model,inferenece_model,criterion_attack,optimizer_mem,epoch,use_cuda)
        print ('test acc',test_acc, at_acc,at_loss,best_acc,att_epoch_acc)
        # append logger file
    

    # save model
        is_best = test_acc>best_acc

        best_acc = max(test_acc, best_acc)


        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best,filename='Depoch%d'%epoch)
        
        
