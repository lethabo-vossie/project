# quarterly_projection_app/models.py
from django.db import models

class QuarterlyData(models.Model):
    quarter = models.CharField(max_length=2)
    year = models.IntegerField()
    sales = models.FloatField()
    expenses = models.FloatField(null=True, blank=True)  # Add this line

    def __str__(self):
        return f"{self.quarter} {self.year} - Sales: {self.sales}, Expenses: {self.expenses}"
