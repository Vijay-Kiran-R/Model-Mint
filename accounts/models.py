from django.db import models
from django.contrib.auth.models import User
from django.contrib.postgres.fields import JSONField  # For Django < 4, use JSONField from pg

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    ganache_address = models.CharField(max_length=100)
    ganache_private_key = models.CharField(max_length=200)
    projects = models.JSONField(default=dict)  # Stores project_name: hash_id

    def __str__(self):
        return self.user.username
