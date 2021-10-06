from django.db import models


class profesional(models.Model):
    id = models.AutoField(primary_key=True)
    tipo_documento = models.CharField(max_length=50)
    no_documento = models.CharField(max_length=100)
    nombres = models.CharField(max_length=100)
    apellidos = models.CharField(max_length=100)
    edad = models.CharField(max_length=50)
    telefono = models.CharField(max_length=100)
    correo = models.EmailField(max_length=200)
    contraseña = models.CharField(max_length=200)

    def __str__(self):
        return self.nombres+" "+self.apellidos


class Paciente(models.Model):
    id = models.AutoField(primary_key=True)
    tipo_documento = models.CharField(max_length=50)
    no_documento = models.CharField(max_length=100)
    nombres = models.CharField(max_length=100)
    apellidos = models.CharField(max_length=100)
    edad = models.CharField(max_length=50)
    telefono = models.CharField(max_length=100)

    def __str__(self):
        return self.nombres+" "+self.apellidos

    def ids(self):
        return self.id


class Cita(models.Model):
    id = models.AutoField(primary_key=True)
    id_profesional = models.ForeignKey(
        profesional, null=False, on_delete=models.CASCADE)
    id_paciente = models.ForeignKey(
        Paciente, null=False, on_delete=models.CASCADE)
    resonancia = models.ImageField(upload_to="resonancias")

    def __str__(self):
        return self.id_paciente


class ImageUpload(models.Model):
    image_file = models.ImageField(upload_to="resonancias")
