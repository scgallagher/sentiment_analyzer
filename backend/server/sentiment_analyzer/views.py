from django.shortcuts import render
from django.http import HttpResponse, HttpResponseBadRequest

def get_topics(request):
    if request.method == 'GET':
        return HttpResponse()
    else:
        return HttpResponseBadRequest()
