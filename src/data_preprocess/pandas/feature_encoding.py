import pandas as pd 
from category_encoders import TargetEncoder
from sklearn import preprocessing

class FeatureEncoder:
    def __init__(self, train_data, test_data, steps):
        self.train_data = train_data
        self.test_data = test_data  
        self.steps = steps 
    
    def process(self):

        for step in self.steps:
            match list(step.keys())[0]: 
                case 'target_encoding': 
                    self.target_encoding(list(step.values())[0])
                case 'label_encoding':
                    self.label_encoding(list(step.values())[0])
        
        return self.train_data, self.test_data  

    def target_encoding(self, params):
        target_col = params['target_col']
        feature_cols = params['feature_cols']
        smoothing = params['smoothing']

        for col in feature_cols:
            tgt_encoder = TargetEncoder(smoothing=smoothing)
            self.train_data[col] = tgt_encoder.fit_transform(self.train_data[col], self.train_data[target_col]).astype('float32')
            self.test_data[col] = tgt_encoder.transform(self.test_data[col], self.test_data[target_col]).astype('float32')

    def label_encoding(self, params):

        feature_cols = params['feature_cols']
        
        for col in feature_cols: 
            label_encoder = preprocessing.LabelEncoder()
            self.train_data[col] = label_encoder.fit_transform(self.train_data[col]).astype('int64')
            self.test_data[col] = label_encoder.transform(self.test_data[col]).astype('int64')

