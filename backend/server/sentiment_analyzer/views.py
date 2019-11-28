from django.shortcuts import render
from django.views import View
from django.http import JsonResponse, HttpResponseBadRequest
from .latent_semantic_analysis import LatentSemanticAnalyzer
import json

class PredictionView(View):

    analyzer = LatentSemanticAnalyzer()
    analyzer.learn()

    def post(self, request):
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        docs = body.get('documents')

        topic_predictions = self.__class__.analyzer.predict_topics(docs)

        return JsonResponse({"topic_predictions": topic_predictions})

    def get(self, request):

        topic_descriptions = self.__class__.analyzer.get_topic_descriptions()
        return JsonResponse({'topics': topic_descriptions})
