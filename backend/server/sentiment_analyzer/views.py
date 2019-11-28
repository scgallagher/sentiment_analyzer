from django.shortcuts import render
from django.views import View
from django.http import HttpResponse, JsonResponse, HttpResponseBadRequest
from .latent_semantic_analysis import LatentSemanticAnalyzer
import json

class PredictionView(View):

    def __init__(self):

        # If analyzer doesn't exist, instantiate it
        try:
            self.__class__.analyzer
        except AttributeError:
            self.__class__.analyzer = LatentSemanticAnalyzer()
            self.__class__.analyzer.learn()

    def post(self, request):
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        docs = body.get('documents')

        if self.__class__.analyzer is None:
            self.__class__.analyzer = LatentSemanticAnalyzer()
            self.__class__.analyzer.learn()
        topic_predictions = self.__class__.analyzer.predict_topics(docs)

        return JsonResponse({"topic_predictions": topic_predictions})
        return HttpResponse()
