from django.shortcuts import render
from .predict import getPrediction

# Create your views here.
def home(request):
    return render(request, 'home.html')

def result(request):
    annual_amount_group_ordinal = float(request.GET['annual_amount_group_ordinal'])
    subscr_month = int(request.GET['subscr_month'])
    age_group_ordinal = float(request.GET['age_group_ordinal'])
    income_mode_5c_ordinal = float(request.GET['income_mode_5c_ordinal'])
    child_filled_mode = float(request.GET['child_filled_mode'])
    fico_mode = float(request.GET['fico_mode'])
    home_value_model = float(request.GET['home_value_model'])
    marital_status_En = float(request.GET['marital_status_En'])
    resid_length_ordinal = float(request.GET['resid_length_ordinal'])
    home_own_freq = float(request.GET['home_own_freq'])
    college_freq = float(request.GET['college_freq'])
    county_Ordinal_Encoding = float(request.GET['county_Ordinal_Encoding'])


    threshold = 0.37

    result = getPrediction(annual_amount_group_ordinal, subscr_month, age_group_ordinal,
                  income_mode_5c_ordinal, child_filled_mode,
                  fico_mode, home_value_model, marital_status_En,
                  resid_length_ordinal, home_own_freq, college_freq,
                  county_Ordinal_Encoding, threshold)
    return render(request, 'result.html', {'result': result})