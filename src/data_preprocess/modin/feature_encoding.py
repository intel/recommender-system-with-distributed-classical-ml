import modin.pandas as pd 
from sklearn import preprocessing
from scipy.special import expit


class TargetEncoder:
    def __init__(self, input_col=None, target_col=None, min_samples_leaf=20, smoothing=10):
        self.input_col = input_col
        self.target_col = target_col
        self.min_samples_leaf = min_samples_leaf 
        self.smoothing = smoothing       
        self.mapping = None
        self._mean = None
        
    def fit(self, X):
        mapping = {}
        
        y = X[self.target_col]
        scalar = self._mean = y.mean()       
        
        if X[self.input_col].dtype.name == 'category':
            X[self.input_col] = X[self.input_col].cat.codes
        
        stats = y.to_frame().groupby(X[self.input_col]).agg({self.target_col:['count', 'mean']})
        stats.columns = stats.columns.droplevel(0)
        
        smoove = self._weighting(stats['count'])
        
        smoothing = scalar * (1-smoove) + stats['mean'] * smoove 
        
        smoothing.loc[-12] = scalar
        mapping[self.input_col] = smoothing 

        return mapping 
    
    def transform(self, X):
        
        if X[self.input_col].dtype.name == 'category':
            result = X[self.input_col].cat.codes.map(self.mapping[self.input_col])
        else:
            result = X[self.input_col].map(self.mapping[self.input_col])
        
        if result.isnull().sum() > 0:
            result.fillna(self.mapping[self.input_col].loc[-12], inplace=True)
            
        return result
    
    
    def fit_transform(self, X):
        
        _X = X.copy()
        self.mapping = self.fit(_X)
        
        return self.transform(X)
    
    def _weighting(self, n):
        tmp = (n - self.min_samples_leaf) / self.smoothing
        res = tmp.apply(lambda sr: expit(sr))
        return res


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
            tgt_encoder = TargetEncoder(input_col=col, target_col=target_col, smoothing=smoothing)
            self.train_data[col] = tgt_encoder.fit_transform(self.train_data)
            self.test_data[col] = tgt_encoder.transform(self.test_data) 

    def label_encoding(self, params):

        feature_cols = params['feature_cols']
        
        for col in feature_cols: 
            label_encoder = preprocessing.LabelEncoder()
            self.train_data[col] = label_encoder.fit_transform(self.train_data[col])
            self.test_data[col] = label_encoder.transform(self.test_data[col])




