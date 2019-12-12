from django.db import models
from django.contrib.postgres.fields import JSONField

class Endpoint(models.Model):

    name = models.CharField(max_length=128)
    owner = models.CharField(max_length=128)
    created_timestamp = models.DateTimeField(auto_now_add=True, blank=True)

class Algorithm(models.Model):

    name = models.CharField(max_length=128)
    description = models.CharField(max_length=1000)
    code = models.CharField(max_length=50000)
    version = models.CharField(max_length=128)
    owner = models.CharField(max_length=128)
    created_timestamp = models.DateTimeField(auto_now_add=True, blank=True)
    parent_endpoint = models.ForeignKey(Endpoint, on_delete=models.CASCADE)

class AlgorithmStatus(models.Model):

    status = models.CharField(max_length=128)
    active = models.BooleanField()
    created_by = models.CharField(max_length=128)
    created_timestamp = models.DateTimeField(auto_now_add=True, blank=True)
    parent_algorithm = models.ForeignKey(Algorithm, on_delete=models.CASCADE)

class ApiRequest(models.Model):

    input_data = JSONField()
    full_response = models.CharField(max_length=10000)
    response = JSONField()
    feedback = JSONField()
    created_timestamp = models.DateTimeField(auto_now_add=True, blank=True)
    parent_algorithm = models.ForeignKey(Algorithm, on_delete=models.CASCADE)
