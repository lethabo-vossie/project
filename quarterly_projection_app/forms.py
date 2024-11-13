# quarterly_projection_app/forms.py
from django import forms

class QuarterlyDataForm(forms.Form):
    sales_data = forms.CharField(
        widget=forms.Textarea(attrs={'placeholder': 'Enter sales data in CSV format (Quarter, Year, Sales)'}),
        # help_text="Enter sales data in CSV format (Quarter, Year, Sales)."
    )
    
    quarters = forms.IntegerField(
        widget=forms.NumberInput(attrs={'placeholder': 'Number of quarters to project'}),
        # help_text="Number of quarters to project."
    )
    graph_type = forms.ChoiceField(
        choices=[('bar', 'Bar'), ('line', 'Line'), ('scatter', 'Scatter')],
        widget=forms.Select(attrs={'placeholder': 'Select the graph type'}),
        # help_text="Select the graph type."
    )
