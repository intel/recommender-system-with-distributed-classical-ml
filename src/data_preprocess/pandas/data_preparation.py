import pandas as pd

class DataPreparator:

    def __init__(self, df, steps):

        self.df = df 
        self.steps = steps 

    def process(self):

        for step in self.steps:
            match list(step.keys())[0]: 
                case 'normalize_feature_names': 
                    self.normalize_feature_names(list(step.values())[0])
                case 'rename_feature_names':
                    raise NotImplementedError 
                case 'drop_features':
                    raise NotImplementedError
                case 'outlier_treatment':
                    raise NotImplementedError
                case 'adjust_datatype':
                    self.adjust_datatype(list(step.values())[0])
        return self.df
            
    def normalize_feature_names(self, steps):
        for step in steps:
            match list(step.keys())[0]:
                case 'replace_chars':
                    self.replace_chars(list(step.values())[0])
                case 'lowercase':
                    if list(step.values())[0]:
                        self.to_lowercase()

    def replace_chars(self, replacements):
        for key, value in replacements.items():
            self.df.columns = self.df.columns.str.replace(key, value)

    def to_lowercase(self):
        self.df.columns = self.df.columns.str.lower()

    def adjust_datatype(self, col_dtypes):
        for col, dtype in col_dtypes.items():
            if isinstance(dtype, list):
                for type in dtype:
                    self.df[col] = self.df[col].astype(type)
            else:
                self.df[col] = self.df[col].astype(dtype)
        
