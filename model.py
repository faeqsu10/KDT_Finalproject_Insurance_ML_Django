import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

import pickle

small_df = pd.read_csv('small_df.csv', index_col=0)

# 데이터 분할
X = small_df[[
    'annual_amount_group_ordinal',
    'subscr_month',
    'age_group_ordinal',
    'income_mode_5c_ordinal',
    'child_filled_mode',
    'fico_mode',
    'home_value_model',
    'marital_status_En',
    'resid_length_ordinal',
    'home_own_freq',
    'college_freq',
    'county_Ordinal_Encoding',
]]
y= small_df['churn']

# 데이터분할(훈련/검증/테스트)
# 먼저 데이터를 훈련 세트와 나머지(검증+테스트)로 분리
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4,
                                                    stratify=y, random_state=42)

# Random UnderSampling 적용
sampler = RandomUnderSampler(random_state=42)
# X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

params = {'bootstrap': False, 'max_depth': 10, 'min_samples_split': 11,
          'min_samples_leaf': 6, 'class_weight': 'balanced', 'n_estimators': 853,
          'max_features': 'sqrt'} #study.best_trial.params

model = RandomForestClassifier(n_jobs=-1, random_state=42, **params)

# 파이프라인 생성
pipe_model = Pipeline([
    ('sampler', sampler),
    ('classifier', model)
])

# 파이프라인을 사용한 훈련
pipe_model.fit(X_train, y_train)

# saving model as a pickle
pickle.dump(pipe_model, open('insurance_rf_model.pkl', 'wb'))