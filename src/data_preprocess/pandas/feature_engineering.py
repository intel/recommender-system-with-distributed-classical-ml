import pandas as pd 
import numpy as np 

class FeatureEngineer:
    def __init__(self, df, steps):
        self.df = df 
        self.steps = steps 
        self.tmp = {}
    
    def process(self):

        for step in self.steps:
            match list(step.keys())[0]: 
                case 'categorify': 
                    self.categorify(list(step.values())[0])
                case 'strip_chars':
                    self.strip_chars(list(step.values())[0])
                case 'combine_cols':
                    self.combine_cols(list(step.values())[0])
                case 'change_datatype':
                    self.change_datatype(list(step.values())[0])
                case 'time_to_seconds':
                    self.time_to_seconds(list(step.values())[0])
                case 'min_max_normalization':
                    self.min_max_normalization(list(step.values())[0])
                case 'one_hot_encoding':
                    self.one_hot_encoding(list(step.values())[0])
                case 'string_to_list':
                    self.string_to_list(list(step.values())[0])
                case 'multi_hot_encoding':
                    self.multi_hot_encoding(list(step.values())[0])
                case 'add_constant_feature':
                    self.add_constant_feature(list(step.values())[0])
                case 'define_variable':
                    self.define_variable(list(step.values())[0])
                case 'modify_on_conditions':
                    self.modify_on_conditions(list(step.values())[0])
        return self.df 

    def categorify(self, features):
        for target_feature, new_feature in features.items():
            self.df[new_feature] = self.df[target_feature].astype('category').cat.codes

    def strip_chars(self, features):
        for old_feature, mapping in features.items():
            for new_feature, char in mapping.items():
                self.df[new_feature] = self.df[old_feature].str.strip(char)
    
    def combine_cols(self, features):
        for new_feature, content in features.items():
            for operation, target_feature_list in content.items(): 
                if len(target_feature_list) < 2:
                    raise ValueError('there is less than 2 items in the list, cannot concatenate')
                else:
                    match operation:
                        case 'concatenate_strings':
                            tmp_feature = self.df[target_feature_list[0]].astype('str')
                            for feature in target_feature_list[1:]:
                                tmp_feature = tmp_feature + self.df[feature].astype('str')
                            self.df[new_feature] = tmp_feature 
    
    def change_datatype(self, col_dtypes):
        for col, dtype in col_dtypes.items():
            self.df[col] = self.df[col].astype(dtype)
    
    def time_to_seconds(self, features):
        for old_feature, new_feature in features.items():
            self.df[old_feature] = self.df[old_feature].astype('datetime64[s]')
            self.df[new_feature] = self.df[old_feature].dt.hour*60 + self.df[old_feature].dt.minute 
    
    def min_max_normalization(self, features):
        for old_feature, new_feature in features.items():
            self.df[new_feature] = (self.df[old_feature] - self.df[old_feature].min())/(self.df[old_feature].max() -  self.df[old_feature].min())
         
    def one_hot_encoding(self, features):
        for feature, is_drop in features.items():
            self.df = pd.concat([self.df, pd.get_dummies(self.df[[feature]])], axis=1)
            if is_drop:
                self.df.drop(columns=[feature], axis=1, inplace=True)

    def string_to_list(self, features):
        for old_feature, mapping in features.items():
            for new_feature, sep in mapping.items():
                self.df[new_feature] = self.df[old_feature].apply(lambda x: str(x).split(sep))
                 
    def multi_hot_encoding(self, features):
        for feature, is_drop in features.items():
            exploded = self.df[feature].explode()
            raw_one_hot = pd.get_dummies(exploded, columns=[feature])
            tmp_df = raw_one_hot.groupby(raw_one_hot.index).sum()
            self.df = pd.concat([self.df, tmp_df], axis=1) 
            col_names = self.df.columns 
            if '' in col_names or 'nan' in col_names: 
                self.df.drop(columns=['', 'nan'], axis=1, inplace=True)
            if is_drop: 
                self.df.drop(columns=[feature], axis=1, inplace=True)
    
    def add_constant_feature(self, features):
        for target_feature, const_value in features.items():
            self.df[target_feature] = const_value

    def define_variable(self, definitions):

        df = self.df
        for var_name, expression in definitions.items():
            self.tmp[var_name] = eval(expression)

    def modify_on_conditions(self, map):
        tmp = self.tmp 
        for col, conditions in map.items():
    
            for condition, value in conditions.items():
                df = self.df
                df.loc[eval(condition), col] = value 
                self.df = df 

