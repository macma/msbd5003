from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.mllib.regression import LabeledPoint
from numpy import array
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from time import time
from operator import attrgetter
sc = SparkContext('local')
# # http://stackoverflow.com/questions/36838024/combining-spark-streaming-mllib


def parse_interaction(line):
    line_split = line.split(",")
    clean_line_split = line_split[0:30]
    label = 1.0
    if line_split[30] == '"0"':
        label = 0.0
    return LabeledPoint(label, array([float(x) for x in clean_line_split]))

def parse_interaction_withoutlabel(line):
    line_split = line.split(",")
    clean_line_split = line_split[0:29]
    label = -1.0
    return LabeledPoint(label, array([float(x) for x in clean_line_split]))


data = sc.textFile('./msbd5003/GroupProject/data/10000.csv')
header = data.first()
data = data.filter(lambda row : row != header)
# train, test = data.randomSplit([0.8, 0.2], seed=12345)
training_data = data.map(parse_interaction)
# testing_data = test.map(parse_interaction)

from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
t0 = time()
model = RandomForest.trainClassifier(training_data, numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=3, featureSubsetStrategy="sqrt",
                                     impurity='entropy', maxDepth=8, maxBins=100)
print('model trained success')

from kafka import KafkaConsumer
consumer = KafkaConsumer('test', bootstrap_servers=['10.0.0.9:9092'])
for msg in consumer:
    print (sc.parallelize([msg.value]).collect())
    streamInput = sc.parallelize([msg.value]).map(parse_interaction_withoutlabel)
    predictions = model.predict(streamInput.map(lambda x: x.features))
    if predictions.collect() > 0:
        print 'the predict result for the input data is: ', predictions.collect()