# Generated by Django 5.1.1 on 2024-11-11 21:06

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='QuarterlyData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('year', models.IntegerField()),
                ('quarter', models.IntegerField(choices=[(1, 'Q1'), (2, 'Q2'), (3, 'Q3'), (4, 'Q4')])),
                ('sales', models.DecimalField(decimal_places=2, max_digits=10)),
                ('expenses', models.DecimalField(decimal_places=2, max_digits=10)),
            ],
        ),
    ]
