
class DataSplitter:
    def __init__(self, df, param):
        self.df = df 
        self.param = param 
    
    def process(self):

        match list(self.param.keys())[0]:
            case 'custom_rules': 
                    self.custom_rules(list(self.param.values())[0])
            case 'random_split':
                    self.random_split(list(self.param.values())[0])
                
        return self.train_data, self.test_data  

    def custom_rules(self, rules):
        train_rules = rules['train']
        test_rules = rules['test']
        df = self.df 
        self.train_data = df[eval(train_rules)]
        self.test_data = df[eval(test_rules)]


    def random_split(self, params):
        pass 



