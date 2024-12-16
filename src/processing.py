"""
Source: https://csyhuang.github.io/2020/08/01/custom-transformer/
"""
from pyspark import keyword_only
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType
import pyspark.sql.functions as F
from itertools import combinations
from pyspark.ml.pipeline import Transformer
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType, FloatType

from math import log


class LogTransformer(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    # Define parameters for input and output columns
    input_cols = Param(Params._dummy(), "input_cols", "input column names.", typeConverter=TypeConverters.toListString)
    output_cols = Param(Params._dummy(), "output_cols", "output column names.", typeConverter=TypeConverters.toListString)

    @keyword_only
    def __init__(self, input_cols=None, output_cols=None):
        super(LogTransformer, self).__init__()
        self._setDefault(input_cols=None, output_cols=None)
        kwargs = self._input_kwargs
        self.set_params(**kwargs)

    @keyword_only
    def set_params(self, input_cols=None, output_cols=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def get_input_cols(self):
        return self.getOrDefault(self.input_cols)

    def get_output_cols(self):
        return self.getOrDefault(self.output_cols)

    def _transform(self, df: DataFrame):
        input_cols = self.get_input_cols()
        output_cols = self.get_output_cols()

        if len(input_cols) != len(output_cols):
            raise ValueError("The number of input columns must match the number of output columns.")

        for input_col, output_col in zip(input_cols, output_cols):
            df = df.withColumn(output_col, F.log(F.col(input_col)))

        return df



class PairwiseTransformer(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):
  inputCols = Param(Params._dummy(), "inputCols", "input column name.", typeConverter = TypeConverters.toList)
  outputCols = Param(Params._dummy(), "outputCols", "output column name.", typeConverter = TypeConverters.toList)
  
  @keyword_only
  def __init__(self, inputCols: list = [], outputCols: list = []):
    super(PairwiseTransformer, self).__init__()
    # Use _set to set the parameter values instead of _setDefault
    kwargs = self._input_kwargs
    self.set_params(**kwargs)
    
  @keyword_only
  def set_params(self, inputCols: list = [], outputCols: list = []):
    # Set parameters using _set
    self._set(inputCols = inputCols, outputCols = outputCols) 
    
  def getInputCols(self):
    return self.getOrDefault(self.inputCols)
  
  def getOutputCols(self):
    return self.getOrDefault(self.outputCols)

  
  def setInputCols(self, value):
        self._set(inputCols=value)
        return self
  
  def setOutputCols(self, value):
      self._set(outputCols=value)
      return self
  
  def _transform(self, df: DataFrame):
    inputCols = self.getInputCols()  # Access input_cols as parameter
    outputCols = self.getOutputCols() # Access output_cols as parameter
    for combination in list(combinations(inputCols, 2)):
      df = df.withColumn(f'{combination[0]}*{combination[1]}', F.col(combination[0]) * F.col(combination[1]))
      df = df.withColumn(f'{combination[0]}+{combination[1]}', F.col(combination[0]) + F.col(combination[1]))
      df = df.withColumn(f'{combination[0]}-{combination[1]}', F.col(combination[0]) - F.col(combination[1]))
    self.setOutputCols([f'{combination[0]}*{combination[1]}' for combination in list(combinations(inputCols, 2))] + [f'{combination[0]}+{combination[1]}' for combination in list(combinations(inputCols, 2))] + [f'{combination[0]}-{combination[1]}' for combination in list(combinations(inputCols, 2))])
    return df



class BoxCoxTransformer(Transformer, HasInputCol, HasOutputCol):
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, alpha=None):
        super(BoxCoxTransformer, self).__init__()
        self.alpha = Param(self, "alpha", 0)
        self._setDefault(alpha=0)
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, alpha=None):
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    def setAlpha(self, value):
        self._paramMap[self.alpha] = value
        return self

    def getAlpha(self):
        return self.getOrDefault(self.alpha)

    def _transform(self, dataset):
        alpha = self.getAlpha()

        def f(s):
            #print(type(s))
            #print(type(alpha))
            if alpha == 0:
                return log(s)
            elif alpha > 0:
                return (s ** alpha - 1) / alpha

        t = FloatType()
        out_col = self.getOutputCol()
        in_col = dataset[self.getInputCol()]
        return dataset.withColumn(out_col, udf(f, t)(in_col))

def calculate_distribution(df, column_name):
    counts = df.groupBy(column_name).count()
    total_count = df.count()
    proportions_df = counts.withColumn(
        "proportion",
        F.col("count") / total_count
    )
    proportions_df.show()