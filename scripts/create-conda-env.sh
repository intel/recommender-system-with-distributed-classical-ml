conda create -n dcml python=3.10 -c anaconda

eval "$(conda shell.bash hook)"

conda activate dcml

pip install --no-cache-dir pyarrow findspark numpy pandas transformers torch pyrecdp scikit-learn category_encoders ray[tune]==2.2.0 xgboost xgboost-ray optuna sigopt pyyaml raydp daal4py simplejson
pip install git+https://github.com/sllynn/spark-xgboost.git
pip install "modin[ray] @ git+https://github.com/dchigarev/modin@fraud_detection_target_enc" 