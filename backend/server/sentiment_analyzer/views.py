from django.shortcuts import render
from django.http import JsonResponse, HttpResponseBadRequest
from .latent_semantic_analysis import LatentSemanticAnalyzer
import json

def __init__(self):

    self.analyzer = LatentSemanticAnalyzer()

def get_topics(request):
    if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        docs = body.get('documents')

        analyzer = LatentSemanticAnalyzer()
        df = analyzer.get_data()
        df = analyzer.clean_data(df)

        analyzer.learn(df)
        topic_predictions = analyzer.predict_topics(docs)

        return JsonResponse({"topic_predictions": topic_predictions})
    else:
        return HttpResponseBadRequest()
