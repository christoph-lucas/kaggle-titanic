import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer, LabelBinarizer, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion

class PipelineLabelBinarizer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.encoder = LabelBinarizer()
        self.encoder.fit(X)
        self.classes_ = self.encoder.classes_
        return self
    def transform(self, X, y = 0):
        return self.encoder.transform(X)

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

class FillNaWith(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_name, replacement='Null'):
        self.replacement = replacement
        self.attribute_name = attribute_name
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_name].fillna(self.replacement)

class Summation(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_name1, attribute_name2, const):
        self.const = const
        self.attribute_name1 = attribute_name1
        self.attribute_name2 = attribute_name2
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.array([X[self.attribute_name1].values + X[self.attribute_name2].values + self.const]).transpose()

class IsAlone(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.array([((X['SibSp'].values + X['Parch'].values) == 0) * 1]).transpose()

class CabinTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = LabelBinarizer();

    def fit(self, X, y=None):
        self.encoder.fit(self.get_cabin_chars(X))
        return self

    def transform(self, X, y=None):
        cabins_tr = []
        # expecting cabin is only column
        cabin_strings = X[:, 0]
        for cabin_string in cabin_strings:
            cabins = cabin_string.split(" ")
            n_cabins = len(cabins)

            try:
                cabin_num = int(cabins[0][1:])
            except:
                cabin_num = -1

            cabins_tr.append([cabin_num, n_cabins])

        cabin_chars = self.get_cabin_chars(X)
        cabin_char_1hot = self.encoder.transform(cabin_chars)
        return np.c_[cabin_char_1hot, cabins_tr]

    def get_cabin_chars(self, X):
        cabin_chars = []
        # expecting cabin is only column
        cabin_strings = X[:, 0]
        for cabin_string in cabin_strings:
            cabins = cabin_string.split(" ")
            try:
                cabin_char = cabins[0][0]
            except IndexError:
                cabin_char = "X"
            cabin_chars.append(cabin_char)

        return cabin_chars

    def get_classes(self):
        return list(self.encoder.classes_) + ["cabin_number", "n_cabins"]

class OrdinalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, ranges):
        self.ranges = ranges
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        res = pd.cut(X[:, 0], self.ranges, labels=list(range(len(self.ranges)-1)))
        return np.array([np.transpose(res.get_values())]).transpose()

class FeatureExtractor:

    def get_attributes(self):
        attributes = ['FamilySize'] + \
                     ['IsAlone'] + \
                     ['AgeCat'] + \
                     ['FareCat'] + \
                     list(self.class_encoder.classes_) + \
                     list(self.sex_encoder.classes_) + \
                     list(self.embarked_encoder.classes_) + \
                     self.cabin_transformer.get_classes()
        return attributes

    def get_feature_union(self):
        family_size_pipeline = Pipeline([
            ('sum', Summation('SibSp', 'Parch', 1)),
        ])

        is_alone_pipeline = Pipeline([
            ('alone', IsAlone()),
        ])

        age_pipeline = Pipeline([
            ('selector', DataFrameSelector(["Age"])),
            ('imputer', Imputer(strategy="median")),
            ('age_ordinal', OrdinalTransformer([0, 16, 32, 48, 64, 100]))
        ])

        fare_pipeline = Pipeline([
            ('selector', DataFrameSelector(["Fare"])),
            ('imputer', Imputer(strategy="median")),
            ('age_ordinal', OrdinalTransformer([-1, 7.91, 14.454, 31, 600]))
        ])

        # cat_attribs = ["Pclass", "Sex", "Embarked"]
        self.class_encoder = PipelineLabelBinarizer()
        class_pipeline = Pipeline([
            ('selector', DataFrameSelector(["Pclass"])),
            ('label_binarizer', self.class_encoder)
        ])

        self.sex_encoder = PipelineLabelBinarizer()
        sex_pipeline = Pipeline([
            ('selector', DataFrameSelector(["Sex"])),
            ('label_binarizer', self.sex_encoder)
        ])

        self.embarked_encoder = PipelineLabelBinarizer()
        embarked_pipeline = Pipeline([
            ('fillna', FillNaWith(["Embarked"])),
            ('selector', DataFrameSelector(["Embarked"])),
            ('label_binarizer', self.embarked_encoder)
        ])

        self.cabin_transformer = CabinTransformer()
        cabin_pipeline = Pipeline([
            ('fillna', FillNaWith(["Cabin"], '')),
            ('selector', DataFrameSelector(["Cabin"])),
            ('cabin_transformer', self.cabin_transformer)
        ])

        combined_features = FeatureUnion(transformer_list=[
            ("family_size_pipeline", family_size_pipeline),
            ("is_alone_pipeline", is_alone_pipeline),
            ("age_pipeline", age_pipeline),
            ("fare_pipeline", fare_pipeline),
            ("class_pipeline", class_pipeline),
            ("sex_pipeline", sex_pipeline),
            ("embarked_pipeline", embarked_pipeline),
            ("cabin_pipeline", cabin_pipeline)
        ])

        return combined_features;
