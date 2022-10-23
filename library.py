import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


class MappingTransformer(BaseEstimator, TransformerMixin):
  
  def __init__(self, mapping_column, mapping_dict:dict):
    assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
    self.mapping_dict = mapping_dict
    self.mapping_column = mapping_column  #column to focus on

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?
    
    #now check to see if all keys are contained in column
    column_set = set(X[self.mapping_column])
    keys_not_found = set(self.mapping_dict.keys()) - column_set
    if keys_not_found:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] these mapping keys do not appear in the column: {keys_not_found}\n")

    #now check to see if some keys are absent
    keys_absent = column_set -  set(self.mapping_dict.keys())
    if keys_absent:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] these values in the column do not contain corresponing mapping keys: {keys_absent}\n")

    X_ = X.copy()
    X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

class OHETransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, dummy_na=False, drop_first=False):  
    self.target_column = target_column
    self.dummy_na = dummy_na
    self.drop_first = drop_first

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.target_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.target_column}"'  
    X_ = X.copy()
    X_ = pd.get_dummies(X_, columns=[self.target_column], dummy_na=self.dummy_na, drop_first=self.drop_first)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
 

class DropColumnsTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column_list, action='drop'):
    assert action in ['keep', 'drop'], f'{self.__class__.__name__} action {action} not in ["keep", "drop"]'
    assert isinstance(column_list, list), f'DropColumnsTransformer expected list but saw {type(column_list)}'
    self.column_list = column_list
    self.action = action

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    remaining_set = set(self.column_list) - set(X.columns)
    # keys_absent = len(set(self.column_list) - set(X.columns)) != 0

    X_ = X.copy()
    if self.action=='drop':
      if remaining_set:
        print(f"\nWarning: {self.__class__.__name__} does not contain these columns to drop: {remaining_set}.")
      X_ = X_.drop(columns=self.column_list, errors='ignore')
    else:
      assert not remaining_set, f'{self.__class__.__name__}.transform unknown columns to keep: {remaining_set}'
      X_ = X_[self.column_list]
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result


class Sigma3Transformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column):  
    self.target_column = target_column
    
  def fit(self, X, y = None):
    print(f"Warning: {self.__class__.__name__}.fit does nothing.")
    return self

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    X_ = X.copy()
    mu = X_[self.target_column].mean()
    sig = X_[self.target_column].std()
    high_bound = mu + 3 * sig
    low_bound = mu - 3 * sig
    X_[self.target_column] = X_[self.target_column].clip(lower=low_bound, upper=high_bound)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

class TukeyTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, fence='outer'):
    assert fence in ['outer', 'inner'] # outer_fence or inner_fence
    self.target_column = target_column
    self.fence = fence
    
  def fit(self, X, y = None):
    print(f"Warning: {self.__class__.__name__}.fit does nothing.")
    return self

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    X_ = X.copy()
    q1 = X_[self.target_column].quantile(0.25)
    q3 = X_[self.target_column].quantile(0.75)
    iqr = q3 - q1
    inner_low = q1 - 1.5 * iqr
    inner_high = q3 + 1.5 * iqr
    outer_low = q1 - 3 * iqr
    outer_high = q3 + 3 * iqr
    
    if self.fence=='inner':
      X_[self.target_column] = X_[self.target_column].clip(lower=inner_low, upper=inner_high)
    else: 
      X_[self.target_column] = X_[self.target_column].clip(lower=outer_low, upper=outer_high)
      
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

  
class MinMaxTransformer(BaseEstimator, TransformerMixin):
  def __init__(self):
    pass #takes no arguments

  #fill in rest below
  def fit(self, X, y = None):
    print(f"Warning: {self.__class__.__name__}.fit does nothing.")
    return X 
    
  def transform(self, X):
    X_ = X.copy()
    scaler = MinMaxScaler()
    fit_transformer = scaler.fit_transform(X_)
    X_ = pd.DataFrame(fit_transformer,columns=X_.columns)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result




 
