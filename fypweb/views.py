from django.http import JsonResponse,HttpResponse
import numpy as np
import pickle
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def home(reques):
  return HttpResponse("<h1 style='display:flex; align-items:center; justify-content:center; width:100vw; font-size:4rem; height:100vh'>Welcome To Final Year Project Api</h1>")

@csrf_exempt
def predict(request):
  if(request.method == 'POST'):
    body = json.loads(request.body)
    query_array = body['query_array']
    query_array = list(map(float,query_array))
    query_array = np.array(query_array).reshape(1,-1)
    response_data={}
    with open("mlmodel\\trainedModel\\trained_model",'rb') as f:
      model = pickle.load(f)
      predict_result = model.predict(query_array)
      response_data['success'] = 'true'
      response_data['predicted_crop'] = predict_result[0]
    return JsonResponse(response_data)
  else:
    return JsonResponse({
      'success': 'false',
      'message':'Invalid request'
    })