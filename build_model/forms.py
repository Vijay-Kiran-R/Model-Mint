from django import forms

class DatasetUploadForm(forms.Form):
    dataset_file = forms.FileField(label='Select a dataset file')
