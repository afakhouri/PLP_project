import sys
import array as pyarray
import warnings

from math import exp, log

from numpy import array, random, tile

from collections import namedtuple

from pyspark import SparkContext, since
from pyspark.rdd import RDD, ignore_unicode_prefix
from pyspark.mllib.common import JavaModelWrapper, callMLlibFunc, callJavaFunc, _py2java, _java2py
from pyspark.mllib.linalg import SparseVector, _convert_to_vector, DenseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.stat.distribution import MultivariateGaussian
from pyspark.mllib.util import Saveable, Loader, inherit_doc, JavaLoader, JavaSaveable
from pyspark.streaming import DStream

class KMeansModel(Saveable, Loader):

    """
    A clustering model derived from the k-means method.
    """

    def __init__(self, centers):
        self.centers = centers
    def clusterCenters(self):
        """Get the cluster centers, represented as a list of NumPy arrays."""
        return self.centers
    def k(self):
        """Total number of clusters."""
        return len(self.centers)
    def predict(self, x):
        """
        Find the cluster that each of the points belongs to in this
        model.

        :param x:
          A data point (or RDD of points) to determine cluster index.
        :return:
          Predicted cluster index or an RDD of predicted cluster indices
          if the input is an RDD.
        """
        best = 0
        best_distance = float("inf")
        if isinstance(x, RDD):
            return x.map(self.predict)

        x = _convert_to_vector(x)
        for i in xrange(len(self.centers)):
            distance = x.squared_distance(self.centers[i])
            if distance < best_distance:
                best = i
                best_distance = distance
        return best
    def computeCost(self, rdd):
        """
        Return the K-means cost (sum of squared distances of points to
        their nearest center) for this model on the given
        data.

        :param rdd:
          The RDD of points to compute the cost on.
        """
        cost = callMLlibFunc("computeCostKmeansModel", rdd.map(_convert_to_vector),
                             [_convert_to_vector(c) for c in self.centers])
        return cost
    def save(self, sc, path):
        """
        Save this model to the given path.
        """
        java_centers = _py2java(sc, [_convert_to_vector(c) for c in self.centers])
        java_model = sc._jvm.org.apache.spark.mllib.clustering.KMeansModel(java_centers)
        java_model.save(sc._jsc.sc(), path)
    def load(cls, sc, path):
        """
        Load a model from the given path.
        """
        java_model = sc._jvm.org.apache.spark.mllib.clustering.KMeansModel.load(sc._jsc.sc(), path)
        return KMeansModel(_java2py(sc, java_model.clusterCenters()))


class KMeans(object):
    def train(cls, rdd, k, maxIterations=100, initializationMode="k-means||",
              seed=None, initializationSteps=2, epsilon=1e-4, initialModel=None):
        """
        Train a k-means clustering model.

        :param rdd:
          Training points as an `RDD` of `Vector` or convertible
          sequence types.
        :param k:
          Number of clusters to create.
        :param maxIterations:
          Maximum number of iterations allowed.
          (default: 100)
        :param initializationMode:
          The initialization algorithm. This can be either "random" or
          "k-means||".
          (default: "k-means||")
        :param seed:
          Random seed value for cluster initialization. Set as None to
          generate seed based on system time.
          (default: None)
        :param initializationSteps:
          Number of steps for the k-means|| initialization mode.
          This is an advanced setting -- the default of 2 is almost
          always enough.
          (default: 2)
        :param epsilon:
          Distance threshold within which a center will be considered to
          have converged. If all centers move less than this Euclidean
          distance, iterations are stopped.
          (default: 1e-4)
        :param initialModel:
          Initial cluster centers can be provided as a KMeansModel object
          rather than using the random or k-means|| initializationModel.
          (default: None)
        """
        clusterInitialModel = []
        if initialModel is not None:
            if not isinstance(initialModel, KMeansModel):
                raise Exception("initialModel is of "+str(type(initialModel))+". It needs "
                                "to be of <type 'KMeansModel'>")
            clusterInitialModel = [_convert_to_vector(c) for c in initialModel.clusterCenters]
        model = callMLlibFunc("trainKMeansModel", rdd.map(_convert_to_vector), k, maxIterations, initializationMode, seed, initializationSteps, epsilon,
                              clusterInitialModel)
        centers = callJavaFunc(rdd.context, model.clusterCenters)
        return KMeansModel([c.toArray() for c in centers])