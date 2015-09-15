__author__ = 'zhuangli'
class MultiLabeledPoint:
    def __init__(self,comment,forward,like,features,dim):
        self.dim=dim
        self.comment=comment
        self.forward=forward
        self.like=like
        self.features=features
