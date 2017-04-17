from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.mllib.regression import LabeledPoint
from numpy import array
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from time import time
sc = SparkContext('local')
spark = SparkSession(sc)

def parse_interaction(line):
    line_split = line.split(",")
    clean_line_split = line_split[0:30]
    label = 1.0
    #print line_split[30]
    if line_split[30]=='"0"':
        label = 0.0
    return LabeledPoint(label, array([float(x) for x in clean_line_split]))
    # else:
    #     lp = []
    #     for num in range(0, 10):
    #         lp.append(LabeledPoint(label, array([float(x) for x in clean_line_split])))
    #     return lp
        

data = sc.textFile('./msbd5003/GroupProject/data/creditcard.csv')
header = data.first()
data = data.filter(lambda row : row != header)
    #.load("data/mllib/sample_linear_regression_data.txt")
train, test = data.randomSplit([0.8, 0.2], seed=12345)
training_data = train.map(parse_interaction)
testing_data = test.map(parse_interaction)

from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
t0 = time()
model = RandomForest.trainClassifier(training_data, numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=3, featureSubsetStrategy="sqrt",
                                     impurity='entropy', maxDepth=8, maxBins=100)
predictions = model.predict(testing_data.map(lambda x: x.features))
labelsAndPredictions = testing_data.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testing_data.count())
print ('Total count:', testing_data.count())
print ('Fraud count in testing data: ', testing_data.filter(lambda lp: lp.label == 1.0).count())
print ('Successfully predited: ', labelsAndPredictions.filter(lambda (v, p): v == p).count())
print ('Failed predited: ', labelsAndPredictions.filter(lambda (v, p): v != p).count(), ". Value: ", labelsAndPredictions.filter(lambda (v, p): v != p).map(lambda (v,p): v).collect())
#print labelsAndPredictions.filter(lambda (v, p): v == p and v == 1.0).map(lambda (v,p):  p).collect()
tt = time() - t0
print('Test Error = ' + str(testErr))
print('Learned classification forest model:')
print ("Total time cost in {} seconds".format(round(tt,3)))
# print(model.toDebugString())

# Naive Bayes requires nonnegative feature values but found 
# from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
# from pyspark.mllib.util import MLUtils
# t0 = time()
# model = NaiveBayes.train(training_data, 1.0)
# tt = time() - t0
# print "Classifier trained in {} seconds".format(round(tt,3))
# # # Make prediction and test accuracy.
# # predictionAndLabel = testing_data.map(lambda p: (model.predict(p.features), p.label))
# # accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / testing_data.count()
# # print('model accuracy {}'.format(accuracy))




# from pyspark.mllib.classification import LogisticRegressionWithLBFGS
# from time import time

# # Build the model
# t0 = time()
# logit_model = LogisticRegressionWithLBFGS.train(training_data)
# tt = time() - t0

# print "Classifier trained in {} seconds".format(round(tt,3))