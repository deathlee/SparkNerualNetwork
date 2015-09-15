__author__ = 'zhuangli'
import numpy as np
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
def sigmoid_prime(x):
    return x*(1.0-x)
def precision(df,dc,dl):
    return (0.2-0.5*df-0.25*dc-0.25*dl)
def deviation(predict,point,mark):
    if mark=="forward":
        target=point.forward
        offset=5
    elif mark=="comment":
        target=point.comment
        offset=3
    elif mark=="like":
        target=point.like
        offset=3
    return np.abs(predict-target)/float(target+offset)
def deviation_prime(predict,point,mark):
    if mark=="forward":
        target=point.forward
        offset=5
    elif mark=="comment":
        target=point.comment
        offset=3
    elif mark=="like":
        target=point.like
        offset=3
    if predict>target:
        return 1.0/float(target+offset)
    elif predict<target:
        return -1.0/float(target+offset)
    else:
        return 0
def dev_p(mark):
    if mark=="forward":
        dev_p=-0.5
    elif mark=="comment":
        dev_p=-0.25
    elif mark=="like":
        dev_p=-0.25
    return dev_p
class NerualNetwork:
    def __init__(self,layers,hiddenLayers="sigmoid",monumentum=0.9,alpha=0.2,epoch=500,batchEpoch=10):
        self.layers=layers
        self.epoch=epoch
        self.batchEpoch=batchEpoch
        if hiddenLayers=="sigmoid":
            self.activation=sigmoid
            self.activation_prime=sigmoid_prime
        self.monumentum=monumentum
        self.alpha=alpha
        self.comment_weight=self.initialWeight(layers)
        self.forward_weight=self.initialWeight(layers)
        self.like_weight=self.initialWeight(layers)
    def initialWeight(self,layers):
        epsilon_in = np.sqrt(6)/np.sqrt(10+layers[1]);
        W_in = np.random.random((layers[0], layers[1]))*2*epsilon_in-epsilon_in
        weight=[]
        weight.append(W_in)
        for i in range(1,len(layers)-1):
            epsilon_h = np.sqrt(6)/np.sqrt(layers[i]+layers[i+1]);
            W_h = np.random.random((layers[i]+1,layers[i+1]))*2*epsilon_h-epsilon_h
            weight.append(W_h)
        return weight
    def forwardPropagate(self,multiLabeledPoint):
        p={}
        pre={}
        for weight,mark in [(self.forward_weight,"forward"),(self.comment_weight,"comment"),(self.like_weight,"like")]:
            output=[multiLabeledPoint.features]
            for l in range(len(weight)-1):
                dot_value = output[l].dot(weight[l])
                activation = self.activation(dot_value)
                output.append(np.append(activation,[1]))
            predict = output[-1].dot(weight[-1].flatten())
            pre[mark]=predict
            output.append(deviation(predict,multiLabeledPoint,mark))
            p[mark]=output
        return precision(p["forward"][-1],p["comment"][-1],p["like"][-1]),p,pre
    def CountFromPartion(self,list_of_lists):
        count=0
        for l in list_of_lists:
            count+=l.comment+l.forward+l.like+1
        return count
    def batchTrain(self,multiLabeledPoints):
        points=list(multiLabeledPoints)
        count=self.CountFromPartion(points)
        gradient={}
        for i in range(self.batchEpoch):
            error=0
            errorCount=0
            c=0
            for point in points:
                c+=1
                precision,p,predict=self.forwardPropagate(point)
                if precision<0:
                    tmpCount=(point.comment+point.forward+point.like+1)
                    divide=float(tmpCount)/float(count)
                    errorCount+=tmpCount
                    self.backPropagate(point,p,divide,gradient,predict)
                    error+=1*divide
        for key in gradient:
            for i in range(len(gradient[key])):
                gradient[key][i]=gradient[key][i]/float(self.batchEpoch)
        return iter((gradient,error,errorCount))
    def choose_weight(self,mark):
        if mark=="forward":
            return self.forward_weight
        elif mark=="comment":
            return self.comment_weight
        elif mark=="like":
            return self.like_weight
    def backPropagate(self,point,p,divide,gradient,predict):
        for key in p:
            weight=self.choose_weight(key)
            output=p[key]
            d_p=dev_p(key)
            delta=[d_p*deviation_prime(predict[key],point,key)]
            for l in range(len(output) - 2, 0, -1):
                delta.append(np.multiply(self.activation_prime(output[l]),delta[-1]*weight[l].T))
            delta.reverse()
            #print delta
            g=[]
            input=output[0].toArray()
            dim=input.shape[0]
            input=input.reshape(dim,1)
            d=np.delete(delta[0],-1)
            g.append(divide*self.alpha *(input).dot(d.reshape(1,d.shape[0])))
            #print delta
            idx=1
            for i in range(1,len(weight)-1):
                g.append(divide*self.alpha * output[i].T.dot(delta[i]))
                i+=1
            #print delta[-1]
            #print output[idx]*delta[-1]
            g.append(divide*self.alpha * output[idx]*delta[-1])
            if key not in gradient:
                gradient[key]=g
            else:
                #print "***********************"
                for i in range(len(gradient[key])):
                    #print len(gradient[key])
                    #print g[i]
                    gradient[key][i]+=g[i]
                #print "***********************"
    def train(self,rdd):
        for i in range(self.epoch):
            resList=rdd.mapPartitions(self.batchTrain)
            sumCount=0
            sumError=0
            res=resList.collect()
            for i in range(0,len(res),3):
                sumCount+=res[i+2]
            for i in range(0,len(res),3):
                gradient=res[i]
                error=res[i+1]
                count=res[i+2]
                for key in gradient:
                    divide=float(count)/float(sumCount)
                    weight=self.choose_weight(key)
                    for i in range(len(weight)):
                        v = np.vstack(gradient[key][i])
                        weight[i]+=divide*v
                sumError+=error*divide
            if sumError<0.1:
                break
            print sumError
    def forwardTest(self,LabeledPoint):
        p={}
        for weight,mark in [(self.forward_weight,"forward"),(self.comment_weight,"comment"),(self.like_weight,"like")]:
            output=[LabeledPoint.features]
            for l in range(len(weight)-1):
                dot_value = output[l].dot(weight[l])
                activation = self.activation(dot_value)
                output.append(np.append(activation,[1]))
            predict = output[-1].dot(weight[-1].flatten())
            p[mark]=predict
        return p
    def predict(self,rdd):
        print "Test Begin:"
        res=rdd.map(self.forwardTest)
        for predict in res.collect():
            print predict["forward"]
            print predict["comment"]
            print predict["like"]