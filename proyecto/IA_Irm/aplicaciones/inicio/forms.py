from django import forms
from django.forms import fields
from .models import profesional, Paciente, Cita, ImageUpload


class PersonaForm(forms.ModelForm):
    class Meta:
        model = profesional
        fields = '__all__'


class pacienteForm(forms.ModelForm):
    class Meta:
        model = Paciente
        fields = '__all__'


class CitaForm(forms.ModelForm):
    class Meta:
        model = Cita
        fields = '__all__'


class ImageForm(forms.ModelForm):
    image_file = forms.ImageField(
        widget=forms.FileInput(attrs={"id":      "image"}))

    class Meta:
        model = ImageUpload
        fields = [
            'image_file',
        ]
