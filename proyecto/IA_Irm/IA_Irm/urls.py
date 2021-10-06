"""IA_Irm URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from aplicaciones.inicio.views import *
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('ia', inicios, name='index'),
    path('menu', modulos, name='menu'),
    path('file-upload', imagenes, name='img'),
    path('files', check_folders, name='file'),
    path('p', pruebas, name='p'),
    path('logs', mensaje, name='logs'),
    path('', login, name='login'),

    path('profesional', vista_profesional, name='profesion'),
    path('profesional_registro', registro_profesional, name='profesion_registro'),
    path('profesional_lista', lista_profesionales, name='listar_registro'),


    path('pacientes', vista_paciente, name='paciente'),
    path('paciente_registro', registro_paciente, name='paciente_registro'),
    path('cita_registro', registro_citas, name='cita_registro'),
    path('paciente_lista', lista_paciente, name='listar_paciente'),
    path('cita_lista', lista_citas, name='listar_cita'),
    path('sesion', logins, name='sesion'),
    path('ia-train', principal_entrenamiento, name='ias'),
    path('cita/<int:id>', analizar, name='show'),
    path('imgs/<str:foto>', mostrar_imagen, name='imgs')


]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)
