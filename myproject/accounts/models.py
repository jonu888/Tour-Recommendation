from django.contrib.auth.models import User
from django.db import models

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    description = models.TextField(blank=True, null=True)
    location= models.CharField(max_length=30, blank=True)
    budjet = models.IntegerField(blank=True, null=True)
    preferred_country = models.CharField(max_length=30, blank=True)

    def __str__(self):
        return self.user.username
