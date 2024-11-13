import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from django.shortcuts import render, redirect
from sklearn.linear_model import LinearRegression
from io import BytesIO
import base64
from .forms import QuarterlyDataForm
from .models import QuarterlyData
from django.http import HttpResponse
import csv
from .forms import QuarterlyDataForm


# Function to simulate seasonality effect with an offset for starting quarter
def add_seasonality(sales, quarters, amplitude=0.1, period=4, start_quarter=1):
    seasonal_effect = amplitude * np.sin(2 * np.pi * (np.arange(len(sales) + quarters) + (start_quarter - 1)) / period)
    return sales + seasonal_effect[:len(sales)], seasonal_effect[len(sales):]

# Function to add random noise to projections for more realism
def add_random_noise(future_sales, noise_level=0.05):
    noise = np.random.normal(0, noise_level * np.mean(future_sales), len(future_sales))
    return future_sales + noise

# View for uploading quarterly data from CSV input
def upload_quarterly_data_view(request):
    if request.method == 'POST':
        form = QuarterlyDataForm(request.POST)
        if form.is_valid():
            csv_data = form.cleaned_data['csv_data']
            csv_reader = csv.reader(csv_data.splitlines())
            next(csv_reader)  # Skip header line

            for row in csv_reader:
                quarter, year, sales = row
                QuarterlyData.objects.create(
                    quarter=quarter,
                    year=int(year),
                    sales=float(sales)
                )
            return redirect('quarterly_projection')  # Redirect to main view after saving

    else:
        form = QuarterlyDataForm()
    return render(request, 'quarterly_projection_app/upload_data.html', {'form': form})

# Main view for handling the quarterly projection
import logging
logger = logging.getLogger(__name__)

def quarterly_projection_view(request):
    if request.method == 'POST':
        form = QuarterlyDataForm(request.POST)
        if form.is_valid():
            sales_data = form.cleaned_data['sales_data']
            expenses_data = form.cleaned_data.get('expenses_data')
            quarters = int(form.cleaned_data['quarters'])
            graph_type = form.cleaned_data['graph_type']

            # Process sales data
            try:
                # Split lines, check for headers, and extract sales values
                sales_lines = sales_data.strip().splitlines()
                if sales_lines[0].startswith("Quarter"):
                    sales_lines = sales_lines[1:]
                sales = np.array([float(line.split(',')[2]) for line in sales_lines])
                
                if len(sales) == 0:
                    raise ValueError("Sales data cannot be empty")
                logger.debug(f"Sales Data: {sales}")
            except ValueError as e:
                logger.error(f"Error in sales data: {e}")
                return render(request, 'quarterly_projection_app/quarterly_projection.html', {
                    'form': form,
                    'error': str(e)
                })

            # Process expenses data if provided
            expenses = None
            if expenses_data:
                try:
                    expenses_lines = expenses_data.strip().splitlines()
                    if expenses_lines[0].startswith("Quarter"):
                        expenses_lines = expenses_lines[1:]
                    expenses = np.array([float(line.split(',')[2]) for line in expenses_lines])

                    if len(expenses) != len(sales):
                        raise ValueError("Expenses data should have the same length as sales data.")
                    logger.debug(f"Expenses Data: {expenses}")
                except ValueError as e:
                    logger.error(f"Error in expenses data: {e}")
                    return render(request, 'quarterly_projection_app/quarterly_projection.html', {
                        'form': form,
                        'error': str(e)
                    })

            # Model fitting and projections (only for sales)
            X = np.arange(1, len(sales) + 1).reshape(-1, 1)
            y = sales.reshape(-1, 1)
            model = LinearRegression()
            try:
                model.fit(X, y)
                logger.debug(f"Model fitted. Coefficients: {model.coef_}")
            except ValueError as e:
                logger.error(f"Model fitting error: {e}")
                return render(request, 'quarterly_projection_app/quarterly_projection.html', {
                    'form': form,
                    'error': f"Model fitting error: {e}"
                })

            # Calculate trend
            slope = model.coef_[0][0]
            trend = "Positive (increasing)" if slope > 0 else "Negative (decreasing)" if slope < 0 else "Stable"

            # Generate future quarters for the projection
            future_X = np.arange(len(sales) + 1, len(sales) + quarters + 1).reshape(-1, 1)
            future_sales = model.predict(future_X).flatten()

            # Process seasonality and noise
            sales_with_seasonality, future_seasonal_effect = add_seasonality(sales, quarters, period=4)
            future_sales_with_seasonality = future_sales + future_seasonal_effect
            future_sales_noisy = add_random_noise(future_sales_with_seasonality)

            # Debugging step to check values before graphing
            logger.debug(f"Sales with seasonality: {sales_with_seasonality}")
            logger.debug(f"Future sales (with noise): {future_sales_noisy}")

            # Generate plot
            fig, ax = plt.subplots()
            create_sales_graph(ax, graph_type, sales_with_seasonality, future_sales_noisy, X, future_X, quarters, "Quarter", expenses)

            # Save plot to a buffer and store it in the session
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image_png = buf.getvalue()
            buf.close()

            graph = base64.b64encode(image_png).decode('utf-8')
            request.session['graph_data'] = graph  # Store graph data in session for downloading
            logger.debug("Graph generated and stored in session.")

            future_sales_list = [(f"Quarter {i}", f"{sale:.2f}") for i, sale in enumerate(future_sales_noisy, start=len(sales) + 1)]

            return render(request, 'quarterly_projection_app/quarterly_projection.html', {
                'form': form,
                'graph': graph,
                'future_sales': future_sales_noisy,
                'future_sales_list': future_sales_list,
                'trend': trend,
            })

    else:
        form = QuarterlyDataForm()

    return render(request, 'quarterly_projection_app/quarterly_projection.html', {'form': form})

# Function to handle the graph type and plot creation
def create_sales_graph(ax, graph_type, sales_with_seasonality, future_sales_noisy, X, future_X, quarters, period_label, expense=None):
    # Plot sales
    if graph_type == 'bar':
        ax.bar(range(1, len(X) + 1), sales_with_seasonality, label='Actual Sales (with seasonality)', color='blue')
        ax.bar(range(len(X) + 1, len(X) + quarters + 1), future_sales_noisy, label=f'{quarters}-{period_label} Projection', color='red')
    elif graph_type == 'line':
        ax.plot(X, sales_with_seasonality, label='Actual Sales (with seasonality)', marker='o')
        ax.plot(future_X, future_sales_noisy, label=f'{quarters}-{period_label} Projection', linestyle='--', marker='x', color='red')
    elif graph_type == 'scatter':
        ax.scatter(X, sales_with_seasonality, label='Actual Sales (with seasonality)', color='blue')
        ax.scatter(future_X, future_sales_noisy, label=f'{quarters}-{period_label} Projection', color='red')

    ax.set(title="Quarterly Sales Projection", xlabel=period_label, ylabel="Amount")
    ax.legend()

# View to download the generated image
def download_graph(request):
    """Serve the graph as a downloadable image from session data."""
    graph_data = request.session.get('graph_data')
    if graph_data:
        image_data = base64.b64decode(graph_data)  # Decode the base64 image data
        response = HttpResponse(image_data, content_type="image/png")
        response['Content-Disposition'] = 'attachment; filename="quarterly_sales_projection.png"'
        return response
    else:
        return HttpResponse("No graph available for download.", status=404)
