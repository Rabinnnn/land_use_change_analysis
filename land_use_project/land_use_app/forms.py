from django import forms

class ImageUploadForm(forms.Form):
    image1 = forms.ImageField(label="First image")
    image2 = forms.ImageField(label="Second image")
