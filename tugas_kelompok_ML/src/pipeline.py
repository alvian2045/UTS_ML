from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(fitur, num_cols, cat_cols):
  num = [c for c in num_cols if c in fitur]
  cat = [c for c in cat_cols if c in fitur]

  num_pipe = Pipeline([
      ('imp', SimpleImputer(strategy='median')),
      ('sc', StandardScaler()),
  ])

  cat_pipe = Pipeline([
      ('imp', SimpleImputer(strategy='most_frequent')),
      (
          'oh',
          OneHotEncoder(handle_unknown='ignore', sparse_output=False),
      ),
  ])

  return ColumnTransformer([
      ('num', num_pipe, num),
      ('cat', cat_pipe, cat),
  ])