# -*- coding: utf-8 -*-
#read this:https://github.com/stephencwelch/Neural-Networks-Demystified/blob/master/Part%202%20Forward%20Propagation.ipynb
"""
Created on Mon Jun 06 14:31:24 2016

@author: amulya
"""
import numpy as np
#ff-feed forward
#l1-first level input
#nh-#of neurons in a hidden layer (assumed only 1 hidden llayer here)
#noIp-#of inputs
#self-keyword used only if something is going to be property of a class/fn.
#eta-learning weight symbol
#itr -how many times being trained
#y-output
class nn:
    def __init__(self,l1,y,nh):
        #setting up the network size (how many hidden layers, ips,ops,etc)
        self.noIp = l1.shape[1] #extracting the # of input neurons (numpy fn)
        self.noY = y.shape[1] #extracting the # of op neurons
        self.noData = l1.shape[0] #extracting the # of objs in study (numpy fn)
        self.nh = nh
        #Weights (parameters)
        self.w1 = (np.random.rand(self.noIp + 1, self.nh))
        print "random w1:", self.w1
        self.w1 = (self.w1-0.5)*2/np.sqrt(self.noIp)    #weights of ips from level 1 ips. # 1-extra 1 for the bias
        print "norm w1:", self.w1        
        self.w2 = (np.random.rand(self.nh + 1, self.noY )-0.5)*2/np.sqrt(self.nh)  #weights of ips from level 2 ips. # 1-extra 1 for the bias
          
    def early_stop_validate(self,l1,y,validateIp, validateTarget, eta, itr=100):
        validateIp=np.concatenate((validateIp,-np.ones((np.shape(validateIp)[0],1))),axis=1) 
        
        old_val_err1 = 100002
        old_val_err2 = 100001
        new_val_err = 100000
        count = 0
        while (((old_val_err1 - new_val_err) > .001) or ((old_val_err2 - old_val_err1) > .001)):
            count +=1
            print count
            self.train_nn(l1, y,eta,itr)
            old_val_err2=old_val_err1
            old_val_err1=new_val_err
            validate_output = self.ff(validateIp)
            new_val_err = 0.5 * np.sum((validateTarget - validate_output)**2)
        print "Stopped ",new_val_err, old_val_err1, old_val_err2
        return new_val_err
            
    def train_nn(self, l1, y,eta,itr):
        #l1 = np.concatenate((np.ones((self.noData,1),l1)),axis = 1)
        l1=np.concatenate((l1,-np.ones((self.noData,1))),axis=1)        
        upd_wt1 = np.zeros((np.shape(self.w1)))
        upd_wt2 = np.zeros((np.shape(self.w2)))
        for n in range(itr):        
            self.opHat = self.ff(l1)
            err = 0.5*np.sum((self.opHat-y)**2)
            if (np.mod(n,100)==0):
                print "iteration: ",n, "error: ",err
            ch_op = (self.opHat - y )*(self.opHat*(1-self.opHat))
            ch_in = self.z*(1-self.z)*(np.dot(ch_op, np.transpose(self.w2)))  #change in layer btw l1 $ cs 
            upd_wt1 = eta*(np.dot(np.transpose(l1),ch_in[:,:-1])) 
            upd_wt2 = eta*(np.dot(np.transpose(self.z),ch_op))
            self.w1 -= upd_wt1
            self.w2 -= upd_wt2         
        
    def ff(self,l1):
         #Propogate inputs though network
        self.z = np.dot(l1 , self.w1)        
        self.z = 1/ (1 + np.exp(-self.z))
        #self.z = np.concatenate((np.ones(self.nh,1)),axis = 1)
        self.z = np.concatenate((self.z,-np.ones((np.shape(l1)[0],1))),axis=1) 
        self.z1 = np.dot(self.z, self.w2)
        self.z1 = 1/ (1 + np.exp(-self.z1)) #Apply sigmoid activation function
        return self.z1
    
    #to check correctness percentage
    def conf_matrix(self,l1,y):
        #l1 = np.concatenate((np.ones(self.noData,1)),axis = 1)
        l1=np.concatenate((l1,-np.ones((np.shape(l1)[0],1))),axis=1)      
        opHat = self.ff(l1)
        clsTot = np.shape(y)[1]
        if clsTot == 1:
            clsTot = 2
            opHat = np.where(opHat > 0.5 , 1 , 0)
        else:
            opHat = np.argmax(opHat, 1)
            y = np.argmax(y , 1)
        conf_mat = np.zeros((clsTot,clsTot))
        for i in range(clsTot):
            for j in range(clsTot):
                conf_mat[i,j] = np.sum(np.where(opHat == i , 1, 0)* np.where(y == j, 1, 0) )
        print "Confusion matrix is:  "
        print conf_mat
        print "% correctness: ", np.trace(conf_mat)/np.sum(conf_mat)*100
        
        
#import nnCls -->calling it from a diff class
#nn1 = nnCls.nn([75,4], [75,1], 3)
        