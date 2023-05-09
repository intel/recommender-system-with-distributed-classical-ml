from sparkxgb import XGBoostClassifier
import os 
import sys 

xgb4j_spark_jar = '/workspace/third_party/xgboost4j-spark_2.12-1.5.2.jar'
xgb4j_jar = '/workspace/third_party/xgboost4j_2.12-1.5.2.jar'
os.environ['PYSPARK_SUBMIT_ARGS'] = f'--jars {xgb4j_spark_jar},{xgb4j_jar} pyspark-shell'


class XGBoostSparkClassifier:

    def __init__(self, xgb_params, label_col):
        self.xgb_params = xgb_params
        self.label_col = label_col 

    def fit(self, data):
        xgb_classifier = XGBoostClassifier(**self.xgb_params).setLabelCol(self.label_col)
        xgb_clf_model = xgb_classifier.fit(data)

        return xgb_clf_model 

        





