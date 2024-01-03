from django.db import models

# Create your models here.
class Course(models.Model):
    username = models.CharField(max_length=255)  # Assuming a maximum length for the username
    course_name = models.CharField(max_length=255)
    course_description = models.TextField()

    def __str__(self):
        return self.course_name

class UploadedFile(models.Model):
    file = models.FileField(upload_to='G:/react-django/django_sessions/uploads')