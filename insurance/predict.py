import pickle
import numpy as np


def getPrediction(annual_amount_group_ordinal, subscr_month, age_group_ordinal,
                  income_mode_5c_ordinal, child_filled_mode,
                  fico_mode, home_value_model, marital_status_En,
                  resid_length_ordinal, home_own_freq, college_freq,
                  county_Ordinal_Encoding,
                  threshold=0.37
                  ):
    model = pickle.load(open('insurance_rf_model.pkl', 'rb'))
    prediction = model.predict([[annual_amount_group_ordinal, subscr_month, age_group_ordinal,
                                 income_mode_5c_ordinal, child_filled_mode,
                                 fico_mode, home_value_model, marital_status_En,
                                 resid_length_ordinal, home_own_freq, college_freq,
                                 county_Ordinal_Encoding]])

    # 예측 확률 계산
    proba = model.predict_proba([[annual_amount_group_ordinal, subscr_month, age_group_ordinal,
                                  income_mode_5c_ordinal, child_filled_mode,
                                  fico_mode, home_value_model, marital_status_En,
                                  resid_length_ordinal, home_own_freq, college_freq,
                                  county_Ordinal_Encoding]])[:, 1]

    # 임계값을 사용하여 예측 결과 결정
    prediction = (proba >= threshold).astype(int)

    print(threshold)

    # result1 = '유지합니다.' if prediction[0] == 0 else '이탈합니다.'
    # 예측 결과 반환
    # return f"고객님은 약 {int(proba[0] * 100)}% 확률로 {result1} "
    return '유지합니다.' if prediction[0] == 0 else '이탈합니다.'
