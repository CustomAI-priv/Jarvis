# tools/charting_tools.py

# Standard library imports
import json
import logging
import random
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple, Union
from urllib.request import urlopen

# Third party imports
import numpy as np
import pandas as pd
import spacy

# NLTK imports
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Plotly imports
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
from plotly.colors import n_colors

# Scipy imports
from scipy import stats, interpolate
from scipy.signal import find_peaks
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.stats import (chi2, chi2_contingency, f_oneway, fisher_exact,
                        gaussian_kde, kendalltau, kruskal, linregress,
                        mannwhitneyu, normaltest, pearsonr, shapiro, spearmanr,
                        ttest_1samp, ttest_ind, ttest_rel, wilcoxon)

# Scikit-learn imports
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score, calinski_harabasz_score,davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import (
    KMeans, AgglomerativeClustering, DBSCAN, 
    SpectralClustering, Birch, MeanShift,
    MiniBatchKMeans, OPTICS
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    HuberRegressor, RANSACRegressor  # Fixed typo here
)
from sklearn.mixture import GaussianMixture

# Statsmodels imports
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import TTestPower
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

# Other imports
from ydata_profiling import ProfileReport


class ToolkitUtilities: 

    def __init__(self):
        # Set the default template
        pio.templates.default = "plotly_white"

        # define the language modeling tools 
        self.nlp = spacy.load('en_core_web_sm')
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.lemmatizer = WordNetLemmatizer()
        
    def apply_template(self, fig):
        """Apply a modern template to the plot with light blue theme and subtle border"""
        fig.update_layout(
            font=dict(family="Arial", size=12),
            title_font=dict(family="Arial", size=24, color="#2171b5"),  # Blue title
            plot_bgcolor="white", 
            paper_bgcolor="white",
            margin=dict(l=80, r=80, t=100, b=80),
            title=dict(x=0.5, xanchor="center"),
            legend=dict(
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(158, 202, 225, 0.8)",  # Light blue border
                borderwidth=1,
                font=dict(size=12),
            ),
            showlegend=True,
        )
        
        # Update bars/markers style with light blue theme
        try: 
          fig.update_traces(
              marker_color='#9ecae1',  # Light blue
              marker_line_color='#6baed6',  # Slightly darker blue for borders
              marker_line_width=1.5,
              opacity=0.8,
          )
        except: 
          pass
        
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(158, 202, 225, 0.4)",  # Very light blue grid
            showline=True,
            linewidth=1,  # Thinner line
            linecolor="#4292c6",  # Medium blue axis line
            title_font=dict(size=14, color="#2171b5"),
            tickfont=dict(size=12, color="#4292c6"),
            nticks=10,
            tickmode='auto',
            tickangle=45,
            title_standoff=20,
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(158, 202, 225, 0.4)",  # Very light blue grid
            showline=True,
            linewidth=1,  # Thinner line
            linecolor="#4292c6",  # Medium blue axis line
            title_font=dict(size=14, color="#2171b5"),
            tickfont=dict(size=12, color="#4292c6"),
            nticks=8,
            tickmode='auto',
            title_standoff=20,
        )
        
        # Add very subtle box around plot
        fig.update_layout(
            shapes=[
                dict(
                    type='rect',
                    xref='paper',
                    yref='paper',
                    x0=0,
                    y0=0,
                    x1=1,
                    y1=1,
                    line=dict(
                        color="rgba(66, 146, 198, 0.3)",  # Transparent medium blue
                        width=1,  # Thinner border
                    ),
                    fillcolor="rgba(255,255,255,0)",
                )
            ]
        )
        
        return fig

    def find_binary_columns(self, data: pd.DataFrame) -> list[str]:
        """
        Find binary columns in a pandas DataFrame.
        """
        return [col for col in data.columns if data[col].nunique() == 2]

    def clean_missing_values_and_outliers(self, data: pd.DataFrame, outlier_method='IQR') -> pd.DataFrame:
        """
        Clean missing values and outliers from a pandas DataFrame.
        
        For each column:
        - Numerical columns: Fill NaN with median for skewed data, mean for normal data
                           Remove outliers using specified method
        - Categorical columns: Fill NaN with mode
                             Preserve duplicates as they are often valid
        
        Parameters:
        - data: pd.DataFrame, the input data
        - outlier_method: str, method to handle outliers ('IQR' or 'confidence_interval')
        
        Returns:
        - pd.DataFrame: Cleaned dataframe
        """
        df = data.copy()
        
        for column in df.columns:
            # Handle missing values based on data type
            if df[column].dtype.kind in 'inf':  # Numerical columns
                # Check if distribution is skewed
                skewness = df[column].skew()
                if abs(skewness) > 1:  # If skewed, use median
                    df[column] = df[column].fillna(df[column].median())
                else:  # If normal-ish, use mean
                    df[column] = df[column].fillna(df[column].mean())
                
                # Remove outliers based on the specified method
                if outlier_method == 'IQR':
                    Q1 = df[column].quantile(0.25)
                    Q3 = df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
                elif outlier_method == 'confidence_interval':
                    mean = df[column].mean()
                    std_dev = df[column].std()
                    lower_bound = mean - 1.96 * std_dev
                    upper_bound = mean + 1.96 * std_dev
                    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
                
            else:  # Categorical columns
                # Fill with mode (most frequent value)
                mode_value = df[column].mode()[0]
                df[column] = df[column].fillna(mode_value)
        
        return df

    def _read_dataframe(self, filename: str, outlier_method='IQR') -> pd.DataFrame:
        """
        Reads a CSV file and returns a pandas DataFrame.
        Cleans missing values and outliers.
        """
        df = pd.read_csv(filename)
        if outlier_method:
            df = self.clean_missing_values_and_outliers(df, outlier_method)
        return df

    @staticmethod
    def cohens_d(group1, group2):
        """
        Calculates Cohen's d effect size between two groups.
        
        Cohen's d measures the standardized difference between two group means.
        Values around:
        0.2 indicate small effect size
        0.5 indicate medium effect size
        0.8 indicate large effect size
        
        Args:
            group1: array-like, first group's data
            group2: array-like, second group's data
            
        Returns:
            float: Cohen's d statistic
        """
        # Get sample sizes
        n1, n2 = len(group1), len(group2)
        
        # Calculate variances with degrees of freedom correction
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Calculate pooled standard error
        pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        # Return Cohen's d statistic
        return (np.mean(group1) - np.mean(group2)) / pooled_se

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop_words]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return ' '.join(tokens)
    
    def line_chart_specific_x_axis_range(self, data: pd.DataFrame, x: str, y: str, start: int = 0, end: int = 1000):
        """
        Create a line chart with a specific x-axis range.
        If x or y is None, finds the range with highest volatility in y values.
        
        Args:
            data: Input DataFrame
            x: Column name for x-axis
            y: Column name for y-axis 
            start: Start of x-axis range
            end: End of x-axis range
            
        Returns:
            DataFrame filtered to specified or most volatile range
        """

        if start is not None and end is not None:
            result = data[(data[x] >= start) & (data[x] <= end)]    
            return result
        else: 
            # Find most volatile range if no range specified
            window_size = len(data) // 10  # Use 10% of data as window
            volatilities = []
            
            for i in range(0, len(data) - window_size):
                window = data.iloc[i:i+window_size]
                volatility = np.std(window[y])
                volatilities.append((i, volatility))
                
            # Get index of window with highest volatility
            max_vol_idx = max(volatilities, key=lambda x: x[1])[0]
            
            return data.iloc[max_vol_idx:max_vol_idx+window_size]


class ChartingTools(ToolkitUtilities):
    """
    A class that provides methods for creating bar plots, line charts,
    histograms, and scatter plots using Plotly with pandas DataFrames.
    """

    def __init__(self):
        super().__init__()

    def save_plot_html(self, fig, filename):
        """
        Saves a Plotly figure as an HTML file.

        Parameters:
        - fig: Plotly Figure object.
        - filename: str, the name of the HTML file to save.
        """
        fig.write_html(filename, full_html=False, include_plotlyjs='cdn')

    def _bar_plot(self, data_filename: str, x: str, y: str, title: str, color: str = None, reduce_bars_condition: bool = False, barmode: str ='group', filename: str ="bar_plot.html"):
        """
        Creates a clean bar plot with grouped data when there are too many points.

        Parameters:
        - data_filename: str, path to the data file
        - x: str, the column name for x-axis data
        - y: str, the column name for y-axis data
        - title: str, the title of the plot
        - color: str, optional column name for color grouping
        - reduce_bars_condition: bool, whether to force bar reduction even with few unique values
        - barmode: str, the bar mode ('group', 'stack', 'overlay', etc.)
        - filename: str, the name of the HTML file to save
        """
        data: pd.DataFrame = self._read_dataframe(data_filename)
        
        # If there are too many unique x values or reduction is forced, group them
        if data[x].nunique() > 50 and reduce_bars_condition:
            # Create bins for x values
            x_min, x_max = data[x].min(), data[x].max()
            bin_width = (x_max - x_min) / 50  # 50 bins for cleaner visualization
            
            # Create bin labels
            bins = np.arange(x_min, x_max + bin_width, bin_width)
            labels = [f"{bins[i]:.0f}-{bins[i+1]:.0f}" for i in range(len(bins)-1)]
            
            # Group data into bins
            data['bin'] = pd.cut(data[x], bins=bins, labels=labels, include_lowest=True)
            
            # Calculate mean for each bin
            grouped = data.groupby('bin', as_index=False).agg({
                y: 'mean',
                x: 'count'  # Count points in each bin
            })
            
            # Create the plot with grouped data
            fig = px.bar(grouped, 
                        x='bin', 
                        y=y,
                        title=title,
                        labels={
                            'bin': x,
                            y: y,
                            'x': 'Count in bin'
                        })
            
            # Update x-axis to show fewer tick labels
            fig.update_xaxes(
                tickmode='array',
                ticktext=labels[::5],  # Show every 5th label
                tickvals=labels[::5],
                tickangle=45
            )
            
        else:
            # If few unique values, use original data
            fig = px.bar(data, x=x, y=y, color=color, barmode=barmode, title=title)
        
        # Apply template and save
        fig = self.apply_template(fig)
        self.save_plot_html(fig, filename)
        return fig

    def _line_plot(self, data_filename: str, x: str, y: str, title: str, color: str = None, specific_x_axis_range: tuple[int, int] = (None, None), filename: str = "line_chart.html", markers: bool = False):
        """
        Creates a line chart and saves it as an HTML file.

        Parameters:
        - data: pandas DataFrame, the input data.
        - x: str, the column name for x-axis data.
        - y: str, the column name for y-axis data.
        - title: str, the title of the plot.
        - color: str, the column name for color grouping.
        - reduce_chart_noise: bool, whether to reduce the noise on the x-axis by grouping the x-axis values into bins of maximum 500 unique values
        - filename: str, the name of the HTML file to save.
        """

        # sort the values of x axis
        data: pd.DataFrame = self._read_dataframe(data_filename)
        data = data.sort_values(by=x)

        # reduce the noise on the x-axis if requested
        if specific_x_axis_range[0] is not None and specific_x_axis_range[1] is not None:
            print('in specifics')
            data = self.line_chart_specific_x_axis_range(data, x, y, specific_x_axis_range[0], specific_x_axis_range[1])

        # create the plot
        fig = px.line(data, x=x, y=y, color=color, title=title, markers=markers)
        fig = self.apply_template(fig)
        self.save_plot_html(fig, filename)
        return fig

    def _histogram_plot(self, data_filename: str, x: str, y: str = None, title: str = None, 
                  nbins: int = 50, color: str = None, filename: str = "histogram.html"):
        """
        Creates a histogram and saves it as an HTML file.

        Parameters:
        - data_filename: str, path to the data file
        - x: str, the column name for the data to be binned
        - y: str, optional column name for y-axis data
        - title: str, optional title of the plot
        - nbins: int, optional number of bins
        - color: str, optional column name for color grouping
        - filename: str, optional name of the HTML file to save
        """
        data: pd.DataFrame = self._read_dataframe(data_filename)
        fig = px.histogram(data, x=x, y=y, nbins=nbins, color=color, title=title)
        fig = self.apply_template(fig)
        self.save_plot_html(fig, filename)
        return fig

    def _scatter_plot(self, data_filename: str, x: str, y: str, title: str = None, 
                    color: str = None, size: str = None, add_regression: bool = False,
                    filename: str = "scatter_plot.html"):
        """
        Creates a scatter plot and saves it as an HTML file.

        Parameters:
        - data_filename: str, path to the data file
        - x: str, the column name for x-axis data
        - y: str, the column name for y-axis data
        - title: str, the title of the plot
        - color: str, optional column name for color grouping
        - size: str, optional column name for marker size variation
        - add_regression: bool, whether to add a regression line
        - filename: str, the name of the HTML file to save
        """
        data: pd.DataFrame = self._read_dataframe(data_filename)
        
        # Create scatter plot first
        fig = px.scatter(data, x=x, y=y, color=color, size=size, title=title)
        
        if add_regression:
            # Add regression line separately to ensure it's on top
            reg_fig = px.scatter(data, x=x, y=y,
                               trendline="ols",
                               trendline_color_override="red")
            
            # Extract only the regression line trace
            reg_trace = next(trace for trace in reg_fig.data if trace.mode == "lines")
            
            # Calculate R² value
            results = px.get_trendline_results(reg_fig)
            r2 = results.px_fit_results.iloc[0].rsquared
            
            # Update regression line properties
            reg_trace.name = f'Regression Line (R² = {r2:.3f})'
            reg_trace.line.width = 2
            reg_trace.line.color = "red"
            
            # Add regression line to original figure
            fig.add_trace(reg_trace)
        
        fig = self.apply_template(fig)
        self.save_plot_html(fig, filename)
        return fig

    def _box_plot(self, data_filename: str, x: str, y: str, title: str, color=None, filename="box_plot.html", group_by_binary: bool=False):
        """
        Creates a box plot and saves it as an HTML file.
        
        For numeric x values:
        - Divides into 2 groups based on median
        
        For categorical x values:
        - Groups into top 5 categories by frequency
        - Remaining categories grouped as 'Other'
        
        Parameters:
        - data_filename: str, path to the data file
        - x: str, column name for x-axis data
        - y: str, column name for y-axis data
        - title: str, plot title
        - color: str, optional column for color grouping
        - filename: str, output filename
        - group_by_binary: bool, whether to use binary columns for grouping
        """
        # read the data
        data: pd.DataFrame = self._read_dataframe(data_filename, outlier_method='confidence_interval')
        
        # Create a copy to avoid modifying original data
        plot_data = data.copy()
        
        if x is not None: 
            # Handle x values based on type and cardinality
            if plot_data[x].nunique() > 2:
                if np.issubdtype(plot_data[x].dtype, np.number):
                    # For numeric columns, split into two groups based on median
                    median = plot_data[x].median()
                    plot_data['grouped_x'] = np.where(
                        plot_data[x] <= median,
                        f'≤ {median:.2f}',
                        f'> {median:.2f}'
                    )
                    # Use the new grouped column for plotting
                    x = 'grouped_x'
                    
                else:
                    # For categorical columns, keep top 5 by frequency percentage
                    value_counts = plot_data[x].value_counts()
                    total_count = len(plot_data)
                    
                    # Calculate percentages
                    percentages = (value_counts / total_count) * 100
                    
                    # Get top 5 categories
                    top_categories = percentages.nlargest(5).index
                    
                    # Group others
                    plot_data['grouped_x'] = plot_data[x].apply(
                        lambda val: val if val in top_categories else 'Other'
                    )
                    
                    # Add percentage to category labels
                    category_percentages = plot_data['grouped_x'].value_counts() / len(plot_data) * 100
                    plot_data['grouped_x'] = plot_data['grouped_x'].apply(
                        lambda x: f"{x} ({category_percentages[x]:.1f}%)"
                    )
                    
                    # Use the new grouped column for plotting
                    x = 'grouped_x'
            
            # Handle binary grouping if requested
            if group_by_binary:
                binary_columns = self.find_binary_columns(data)
                if binary_columns:
                    binary_column = random.choice(binary_columns)
                    color = binary_column
        
        # Create the plot
        fig = px.box(plot_data, x=x, y=y, color=color, title=title)
        
        # Update layout for better readability
        fig.update_xaxes(
            title=title
        )
        
        # Add hover data showing original x values if grouped
        if x != 'grouped_x'and x is not None:
            fig.update_traces(
                hovertemplate=(
                    f"{x}: %{{customdata}}<br>" +
                    f"{y}: %{{y}}<br>" +
                    "<extra></extra>"
                ),
                customdata=plot_data[x]
            )
        
        fig = self.apply_template(fig)
        self.save_plot_html(fig, filename)
        return fig

    def _dist_plot(self, data_filename: str, x: str, y: str, title: str, color=None, marginal: str = 'violin', filename="dist_plot.html"):
        """
        Creates a distribution plot and saves it as an HTML file.
        The marginal parameter can be 'box' or 'violin'
        """

        # read the data
        data: pd.DataFrame = self._read_dataframe(data_filename)

        # build the plot
        fig = px.histogram(data, x=x, y=y, color=color, marginal=marginal,
                   hover_data=data.columns, title=title)
        fig = self.apply_template(fig)
        self.save_plot_html(fig, filename)
        return fig
    
    def multiple_dist_plot(self, data_filenames: list[str], x: list[str], title: str, color=None, filename="multiple_dist_plot.html"):
        """
        Creates a multiple distribution plot and saves it as an HTML file.
        """

        # read the data
        data: list[pd.DataFrame] = [self._read_dataframe(df) for df in data_filenames]

        # build the plot
        hist_data = [df[x].values for df in data]
        group_labels = [df.name for df in data]

        # Create distplot with custom bin_size
        fig = ff.create_distplot(hist_data, group_labels, bin_size=.2, title=title, colors=color)
        fig = self.apply_template(fig)
        self.save_plot_html(fig, filename)
        return fig

    def _density_heatmap_plot(self, data_filename: str, x: str, y: str, title: str, color=None, marginal: str = None, filename="density_heatmap.html"):
        """
        Creates a density heatmap plot and saves it as an HTML file.
        """
        # read the data
        data: pd.DataFrame = self._read_dataframe(data_filename)

        # build the plot
        if marginal:
            fig = px.density_heatmap(data, x=x, y=y, marginal_x=marginal, marginal_y=marginal, title=title)
        else:
            fig = px.density_heatmap(data, x=x, y=y, title=title)
        fig = self.apply_template(fig)
        self.save_plot_html(fig, filename)
        return fig

    def error_bars_plot(self, data_filename: str, x: str, y: str, title: str, color=None, filename="error_bars_plot.html"):
        """
        Creates a error bars plot and saves it as an HTML file.
        """
        
        # read the data
        data: pd.DataFrame = self._read_dataframe(data_filename)

        # add the error column to the dataframe by 1 percent of the value
        data['ex'] = data[x] * 0.01
        data['ey'] = data[y] * 0.01

        # build the plot
        fig = px.scatter(data, x=x, y=y, color=color,
                 error_x="ex", error_y="ey")
        fig = self.apply_template(fig)
        self.save_plot_html(fig, filename)
        return fig
    
    def pie_plot(self, data_filename: str, values: str, labels: str, title: str, filename="pie_chart.html"):
        """
        Creates a pie chart and saves it as an HTML file.
        Groups smaller categories into "Other" based on value distribution.
        
        Parameters:
        - data_filename: str, path to the data file
        - values: str, column name for values
        - labels: str, column name for category labels
        - title: str, chart title
        - filename: str, output filename
        """
        # Read the data
        data: pd.DataFrame = self._read_dataframe(data_filename)
        
        # Calculate threshold using mean and standard deviation
        threshold = data[values].mean() * 0.25  # You can adjust this multiplier
        
        # Create a copy to avoid modifying original data
        plot_data = data.copy()
        
        # Rename smaller categories to "Other"
        plot_data.loc[plot_data[values] < threshold, labels] = 'Other'
        
        # Sort values in descending order for better visualization
        plot_data = plot_data.sort_values(values, ascending=False)

        # Build the plot
        fig = px.pie(plot_data, 
                    values=values, 
                    names=labels,
                    title=title,
                    hover_data=[values], 
                    color_discrete_sequence=px.colors.sequential.YlGnBu)
        
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            hovertemplate="<b>%{label}</b><br>" +
                         f"{values}: %{{value:,.0f}}<extra></extra>"
        )

        fig = self.apply_template(fig)
        self.save_plot_html(fig, filename)
        return fig

    def tree_plot(self, data_filename: str, x: str, y: str, title: str, filename="tree_map.html"):
        """
        Creates a tree map and saves it as an HTML file.
        """

        # read the data
        data: pd.DataFrame = self._read_dataframe(data_filename)

        # build the plot
        nodes_data: pd.DataFrame = data[[x, y]]
        nodes_data['id'] = [i for i in range(nodes_data.shape[0])]

        # Create the edge coordinates lists
        edge_x = []
        edge_y = []
        for _, edge in nodes_data.iterrows():
            source = nodes_data[nodes_data['id'] == edge['source']]
            target = nodes_data[nodes_data['id'] == edge['target']]
            edge_x += [source['x'].values[0], target['x'].values[0], None]
            edge_y += [source['y'].values[0], target['y'].values[0], None]

        # Create the node trace
        node_trace = go.Scatter(
            x=nodes_data['x'],
            y=nodes_data['y'],
            mode='markers+text',
            text=nodes_data['id'],
            textposition="bottom center",
            marker=dict(size=10, color="skyblue", line=dict(width=1)),
            hoverinfo="text"
        )

        # Create the edge trace
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color="black"),
            hoverinfo="none",
            mode='lines'
        )

        # Combine both traces in a single figure
        fig = go.Figure(data=[edge_trace, node_trace], title=title)

        # Add layout options
        fig.update_layout(
            title="Tree Graph Visualization",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            hovermode='closest'
        )

        self.save_plot_html(fig, filename)
        return fig

    def violin_plot(self, data_filename: str, y_vars: list[str], title: str, 
                   group_by: str = None, split: bool = False, 
                   show_box: bool = True, show_points: str = 'all',
                   filename: str = "violin_plot.html"):
        """
        Creates violin plots with basic or advanced grouping options.
        
        Parameters:
        - data_filename: str, path to the data file
        - y_vars: list[str], list of variables to plot
        - title: str, chart title
        - group_by: str, optional column to group by (for advanced plotting)
        - split: bool, whether to split violins for grouped data (default False)
        - show_box: bool, whether to show box plot inside violin
        - show_points: str, 'all', 'outliers', or False
        - filename: str, output filename
        
        Examples:
        # Basic multiple violin plots
        violin_plot(data, ['salary', 'age', 'experience'], 'Distribution Metrics')
        
        # Grouped violin plot by category
        violin_plot(data, ['salary'], 'Salary by Department', group_by='department')
        
        # Split violin plot for binary comparisons
        violin_plot(data, ['salary'], 'Salary by Gender', group_by='gender', split=True)
        """
        # Read the data
        data: pd.DataFrame = self._read_dataframe(data_filename)
        
        if group_by is None:
            # Basic violin plots for multiple variables
            fig = go.Figure()
            for var in y_vars:
                fig.add_trace(go.Violin(
                    y=data[var],
                    name=var,
                    box_visible=show_box,
                    meanline_visible=True,
                    points=show_points,
                    jitter=0.05,  # Add jitter for better point visibility
                    scalemode='count'  # Scale violin area with total count
                ))
        else:
            # Advanced grouped violin plots
            fig = go.Figure()
            groups = data[group_by].unique()
            
            if split and len(groups) == 2:  # Split only works well with binary variables
                colors = ['lightseagreen', 'mediumpurple']  # Custom colors for split
                for idx, group in enumerate(groups):
                    fig.add_trace(go.Violin(
                        x=data[group_by][data[group_by] == group],
                        y=data[y_vars[0]][data[group_by] == group],
                        legendgroup=str(group),
                        scalegroup=str(group),
                        name=str(group),
                        side='negative' if idx == 0 else 'positive',
                        line_color=colors[idx],
                        points=show_points,
                        jitter=0.05,
                        scalemode='count',
                        box_visible=show_box,
                        meanline_visible=True
                    ))
                fig.update_layout(violingap=0, violinmode='overlay')
                
            else:
                # Regular grouped violin plot
                for group in groups:
                    fig.add_trace(go.Violin(
                        x=data[group_by][data[group_by] == group],
                        y=data[y_vars[0]][data[group_by] == group],
                        name=str(group),
                        box_visible=show_box,
                        meanline_visible=True,
                        points=show_points,
                        jitter=0.05,
                        scalemode='count'
                    ))
                fig.update_layout(violinmode='group')
        
        # Update layout with additional styling
        fig.update_layout(
            title_text=f"{title}<br><i>scaled by count</i>",
            showlegend=True,
            violingap=0.1,  # Gap between violins
            violingroupgap=0.05  # Gap between violin groups
        )
        
        # apply the template
        fig = self.apply_template(fig)
        self.save_plot_html(fig, filename)
        return fig

    def ridgeline_plot(self, data_filename: str, list_of_columns: list[str], title: str, filename="ridgeline_plot.html"):
        """
        Creates a ridgeline plot and saves it as an HTML file.
        """

        # read the data
        data: pd.DataFrame = self._read_dataframe(data_filename)

        # subset the data
        data = data[list_of_columns]

        # build the plot
        colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', len(data.shape[1]), colortype='rgb')

        fig = go.Figure(title=title)
        for data_line, color in zip(data, colors):
            fig.add_trace(go.Violin(x=data_line, line_color=color))

        fig.update_traces(orientation='h', side='positive', width=3, points=False)
        fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)

        # apply the template
        fig = self.apply_template(fig)
        self.save_plot_html(fig, filename)
        return fig

    def table_plot(self, data_filename: str, list_of_columns: list[str], title: str, filename="table_plot.html"):
        """
        Creates a table plot and saves it as an HTML file.
        """

        # read the data
        data: pd.DataFrame = self._read_dataframe(data_filename)

        # subset the data
        data = data[list_of_columns]

        # build the plot    
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(data.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=list(data.values),
                    fill_color='lavender',
                    align='left'))
        ], title=title)

        # apply the template
        fig = self.apply_template(fig)
        self.save_plot_html(fig, filename)
        return fig

    def histogram_contour_plot(self, data_filename: str, x: str, y: str, facet_column: str = None, binary_col: str = None, title: str = None, filename: str = "histogram_contour_plot.html"):
        """
        Creates a histogram contour plot and saves it as an HTML file.
        """

        # read the data
        data: pd.DataFrame = self._read_dataframe(data_filename)

        # check if a binary column is provided
        if binary_col:
            fig = px.density_contour(data, x=x, y=y, facet_col=facet_column, color=binary_col, title=title)
            fig = self.apply_template(fig)
        else:
            fig = go.Figure(go.Histogram2dContour(
                    x = x,
                    y = y,
                    colorscale = 'Blues'
            ))

        # save the plot
        self.save_plot_html(fig, filename)
        return fig
    
    def contour_plot(self, data_filename: str, list_of_columns: list = [], title: str = None, filename: str = "contour_plot.html"):
        """
        Creates a contour plot and saves it as an HTML file.
        """

        # read the data
        data: pd.DataFrame = self._read_dataframe(data_filename)

        # build the plot
        fig = go.Figure(data =
            go.Contour(
                z=[data[col].tolist() for col in list_of_columns]
            ))
        fig = self.apply_template(fig)
        self.save_plot_html(fig, filename)
        return fig
    
    def dot_plot(self, data_filename: str, x: str, y: str, title: str, filename: str = "dot_plot.html"):
        """
        Creates a dot plot and saves it as an HTML file.
        """       

        # read the data
        data: pd.DataFrame = self._read_dataframe(data_filename)

        # build the plot
        fig = px.dot(data, x=x, y=y, title=title)
        fig = self.apply_template(fig)
        self.save_plot_html(fig, filename)
        return fig
    
    def heatmap_plot(self, data_filename: str, x: str, y: str, z: str, title: str = None, filename: str = "heatmap_plot.html"):
        """
        Creates a heatmap plot and saves it as an HTML file.
        """ 
        
        # read the data
        data: pd.DataFrame = self._read_dataframe(data_filename)

        # build the plot    
        # Calculate total counts and percentages
        total = data[z].sum()
        value_counts = data[z].value_counts()
        percentages = value_counts / total * 100

        # Group small categories into "Other"
        mask = percentages < 5
        other_sum = value_counts[mask].sum()
        
        # Create new grouped data
        grouped_data = value_counts[~mask].copy()
        if other_sum > 0:
            grouped_data['Other'] = other_sum

        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=[grouped_data.values],
            x=grouped_data[x],
            y=grouped_data[y],
            colorscale='Blues', 
            title=title
        ))

        # apply the template
        fig = self.apply_template(fig)
        self.save_plot_html(fig, filename)
        return fig

    def candlestick_plot(self, data_filename: str, open_column: str, high_column: str, low_column: str, close_column: str, title: str, filename: str = "candlestick_plot.html"):
        """
        Creates a candlestick plot and saves it as an HTML file.
        """

        # read the data
        data: pd.DataFrame = self._read_dataframe(data_filename)

        # Check for any datetime columns
        date_columns = data.select_dtypes(include=['datetime64']).columns
        
        # Also check for columns that could be dates but aren't converted yet
        potential_date_cols = []
        for col in data.columns:
            try:
                pd.to_datetime(data[col])
                potential_date_cols.append(col)
            except:
                continue
                
        if len(date_columns) == 0 and len(potential_date_cols) == 0:
            raise ValueError("No date columns found in the data. Please provide data with a datetime column.")
        
        # If we found potential date columns but they're not converted, convert the first one
        if len(date_columns) == 0 and len(potential_date_cols) > 0:
            data[potential_date_cols[0]] = pd.to_datetime(data[potential_date_cols[0]])
            date_columns = [potential_date_cols[0]]
            
        # Use the first date column found
        date_column = date_columns[0]

        # build the plot
        fig = go.Figure(data=[go.Candlestick(x=data[date_column],
                open=data[open_column],
                high=data[high_column],
                low=data[low_column],
                close=data[close_column])])

        # apply the template
        fig = self.apply_template(fig)
        self.save_plot_html(fig, filename)
        return fig

    def facet_timeseries_plot(self, data_filenames: list[str], facet_column: str, title: str, filename: str = "facet_timeseries_plot.html"):
        """
        Creates a facet timeseries plot and saves it as an HTML file.
        """

        # read the data
        data: list[pd.DataFrame] = [self._read_dataframe(df) for df in data_filenames]

        # Check for any datetime columns
        date_columns = data.select_dtypes(include=['datetime64']).columns
        
        # Also check for columns that could be dates but aren't converted yet
        potential_date_cols = []
        for col in data.columns:
            try:
                pd.to_datetime(data[col])
                potential_date_cols.append(col)
            except:
                continue
                
        if len(date_columns) == 0 and len(potential_date_cols) == 0:
            raise ValueError("No date columns found in the data. Please provide data with a datetime column.")
        
        # If we found potential date columns but they're not converted, convert the first one
        if len(date_columns) == 0 and len(potential_date_cols) > 0:
            data[potential_date_cols[0]] = pd.to_datetime(data[potential_date_cols[0]])
            date_columns = [potential_date_cols[0]]
            
        # Use the first date column found
        date_column = date_columns[0]

        # Melt the DataFrame to long format
        df_long = data.melt(id_vars=date_column, var_name='melted_var', value_name='value')

        # Plot with Plotly
        fig = px.area(
            df_long, 
            x=date_column, 
            y='value', 
            color='melted_var',  # Optional, but adds a color legend for company
            facet_col=facet_column, 
            facet_col_wrap=2  # Creates a 3x2 layout automatically
        )

        # Update layout for aesthetics
        fig.update_layout(
            title="Time Series Area Chart for Multiple Companies",
            showlegend=False,  # Hides legend to reduce redundancy since each subplot is labeled
        )

        # Ensure consistent x- and y-axis formatting
        fig.update_xaxes(matches='x')  # Aligns all x-axes
        fig.update_yaxes(showgrid=False)  # Optionally hides gridlines for a cleaner look

        # apply the template
        fig = self.apply_template(fig)
        self.save_plot_html(fig, filename)
        return fig

    def timeseries_plot(self, data_filename: str, y: str, title: str, filename: str = "timeseries_plot.html"):
        """
        Creates a timeseries plot and saves it as an HTML file.
        """

        # read the data
        data: pd.DataFrame = self._read_dataframe(data_filename)

        # Check for any datetime columns
        date_columns = data.select_dtypes(include=['datetime64']).columns
        
        # Also check for columns that could be dates but aren't converted yet
        potential_date_cols = []
        for col in data.columns:
            try:
                pd.to_datetime(data[col])
                potential_date_cols.append(col)
            except:
                continue
                
        if len(date_columns) == 0 and len(potential_date_cols) == 0:
            raise ValueError("No date columns found in the data. Please provide data with a datetime column.")
        
        # If we found potential date columns but they're not converted, convert the first one
        if len(date_columns) == 0 and len(potential_date_cols) > 0:
            data[potential_date_cols[0]] = pd.to_datetime(data[potential_date_cols[0]])
            date_columns = [potential_date_cols[0]]
            
        # Use the first date column found
        date_column = date_columns[0]

        # build the plot
        fig = px.line(data, hover_data={"date": "|%B %d, %Y"}, x=date_column, y=y, title=title)

        # add a range slider to the plot 
        fig.update_xaxes(rangeslider_visible=True)

        fig = self.apply_template(fig)
        self.save_plot_html(fig, filename)
        return fig
    
    def radar_plot(self, data_filename: str, list_of_columns: list[str], y: str, title: str, filename: str = "radar_plot.html"):
        """
        Creates a radar plot and saves it as an HTML file.
        """

        # read the data
        data: pd.DataFrame = self._read_dataframe(data_filename)

        # Calculate percentages for column y
        total = data[y].sum()
        value_counts = data[y].value_counts()
        percentages = (value_counts / total) * 100
        
        # Create a copy of the data
        plot_data = data.copy()
        
        # Get categories that are less than 5%
        small_categories = value_counts[percentages < 5].index
        
        # Rename all small categories to "Other"
        plot_data.loc[plot_data[y].isin(small_categories), y] = 'Other'

        # build the plot
        fig = go.Figure()

        for col in list_of_columns:
            fig.add_trace(go.Scatterpolar(
                r=plot_data[col],
                theta=plot_data[y],
                fill='toself',
                name=col
            ))

        # update the layout of the plot 
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                visible=True,
                range=[0, 5]
                )),
            showlegend=False
        )

        # apply the template
        fig = self.apply_template(fig)
        self.save_plot_html(fig, filename)
        return fig

    def tile_chloropleth_plot(self, data_filename: str, x: str, title: str, filename: str = "tile_chloropleth_plot.html"):
        """
        Creates a tile chloropleth plot and saves it as an HTML file.
        """

        # read the geographic data 
        with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
            counties = json.load(response)
        
        # read the data
        data: pd.DataFrame = self._read_dataframe(data_filename)

        # check if fips is in the data 
        if 'fips' not in data.columns:
            raise ValueError("fips column not found in the data. location is not found")

        # build the plot
        fig = px.choropleth(data, geojson=counties, locations='fips', color=x,
                           color_continuous_scale="Blues",
                           range_color=(0, 12),
                           scope="usa",
                           #labels={'unemp':'unemployment rate'}, 
                           title=title, 
                        )
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

        # apply the template
        #fig = self.apply_template(fig)
        self.save_plot_html(fig, filename)
        return fig
    

class StatisticalTools(ToolkitUtilities):
    def __init__(self):
        super().__init__()

    def correlation_matrix(self, data_filename: str, correlation_variables: list[str], title: str, filename: str = "correlation_matrix.html"):
        """
        Creates a correlation matrix and saves it as an HTML file.
        """ 

        # read the data
        data: pd.DataFrame = self._read_dataframe(data_filename)

        # build the plot
        fig = px.imshow(data[correlation_variables].corr(), title=title)

        # apply the template
        fig = self.apply_template(fig)
        self.save_plot_html(fig, filename)

    def calculate_correlation(self, data_filename: str, x: str, y: str):
        """
        Automatically selects and calculates the appropriate correlation method
        based on data characteristics.
        
        Selection Logic:
        - Pearson: For continuous numerical variables with linear relationship
        - Spearman: For ordinal variables or monotonic relationships
        - Kendall: For ordinal variables with small sample size or ties
        - Phi_K: For mixed-type variables
        - Cramer's V: For categorical variables with more than 2 categories
        
        Returns:
        - tuple: (correlation_value, method_used)
        """
        # Read the data
        data: pd.DataFrame = self._read_dataframe(data_filename)
        
        def is_numeric(series):
            return pd.api.types.is_numeric_dtype(series)
        
        def is_categorical(series):
            return pd.api.types.is_categorical_dtype(series) or series.nunique() < 10
        
        def has_normal_distribution(series):
            _, p_value = stats.normaltest(series)
            return p_value > 0.05
        
        x_data, y_data = data[x], data[y]
        sample_size = len(data)
        
        # Check data types and characteristics
        if is_numeric(x_data) and is_numeric(y_data):
            # Both variables are numeric
            if sample_size < 30:
                # Small sample size: use Kendall's Tau
                correlation, _ = kendalltau(x_data, y_data)
                method = 'kendall'
            elif has_normal_distribution(x_data) and has_normal_distribution(y_data):
                # Normal distribution: use Pearson
                correlation, _ = pearsonr(x_data, y_data)
                method = 'pearson'
            else:
                # Non-normal distribution: use Spearman
                correlation, _ = spearmanr(x_data, y_data)
                method = 'spearman'
                
        elif is_categorical(x_data) and is_categorical(y_data):
            # Both variables are categorical
            
            contingency_table = pd.crosstab(x_data, y_data)
            
            if (x_data.nunique() == 2) and (y_data.nunique() == 2):
                # 2x2 categorical: use Phi coefficient
                chi2, _, _, _ = chi2_contingency(contingency_table)
                n = contingency_table.sum().sum()
                correlation = np.sqrt(chi2 / n)
                method = 'phi'
            else:
                # Multi-category: use Cramer's V
                chi2, _, _, _ = chi2_contingency(contingency_table)
                n = contingency_table.sum().sum()
                min_dim = min(contingency_table.shape) - 1
                correlation = np.sqrt(chi2 / (n * min_dim))
                method = 'cramers_v'
                
        else:
            # Mixed types: use Phi K correlation
            if is_numeric(x_data):
                # Convert numeric to categorical for mixed-type analysis
                x_data = pd.qcut(x_data, q=5, duplicates='drop')
            if is_numeric(y_data):
                y_data = pd.qcut(y_data, q=5, duplicates='drop')
                
            contingency_table = pd.crosstab(x_data, y_data)
            chi2, _, _, _ = chi2_contingency(contingency_table)
            n = contingency_table.sum().sum()
            min_dim = min(contingency_table.shape) - 1
            correlation = np.sqrt(chi2 / (n * min_dim))
            method = 'phi_k'
        
        return correlation, method

    def descriptive_statistics(self, data_filename: str, filename: str = "descriptive_statistics.html"):
        """
        Creates a descriptive statistics report and saves it as an HTML file.
        """

        # read the data
        data: pd.DataFrame = self._read_dataframe(data_filename)

        # build the report
        report = ProfileReport(data, title="Descriptive Statistics")

        # save the report
        report.to_file(filename)
    
    def covariance_matrix(self, data_filename: str, filename: str = "covariance_matrix.html"):
        """
        Creates a covariance matrix and saves it as an HTML file.
        """

        # read the data
        data: pd.DataFrame = self._read_dataframe(data_filename)

        # build the plot
        fig = px.imshow(data.cov(), title="Covariance Matrix")

        # apply the template
        fig = self.apply_template(fig)
        self.save_plot_html(fig, filename)
    
    def calculate_covariance(self, data_filename: str, x: str, y: str):
        """
        Calculates the covariance between two variables.
        """

        # read the data
        data: pd.DataFrame = self._read_dataframe(data_filename)

        # calculate the covariance
        covariance = data[x].cov(data[y])

        return covariance

    def perform_t_test(self, data_filename: str, x: str, y: str):
        """
        Performs a t-test between two variables.
        """

        # read the data
        data: pd.DataFrame = self._read_dataframe(data_filename)

        # perform the t-test
        t_stat, p_value = ttest_ind(data[x], data[y])

        return t_stat, p_value

    def perform_f_test(self, data_filename: str, x: str, y: str):
        """
        Performs an F-test between two variables.
        """

        # read the data
        data: pd.DataFrame = self._read_dataframe(data_filename)

        # perform the f-test
        f_stat, p_value = f_oneway(data[x], data[y])

        return f_stat, p_value

    def calculate_p_value(self, data_filename: str, x: str, y: str):
        """
        Calculates the p-value between two variables.
        """

        # read the data
        data: pd.DataFrame = self._read_dataframe(data_filename)

        # calculate the p-value
        p_value = data[x].pvalue(data[y])

        return p_value

    def normality_test(self, data_filename: str, x: str):
        """
        Performs normality tests on a variable.
        
        Returns a dictionary containing test statistics and p-values for both
        D'Agostino-Pearson and Shapiro-Wilk tests. D'Agostino-Pearson is recommended
        for general use but Shapiro-Wilk may be better when values are unique.
        """
        # read the data 
        data: pd.DataFrame = self._read_dataframe(data_filename)
        
        # Get the column data
        column_data = data[x]
        
        # For sample sizes < 5000, run both tests
        if len(column_data) < 5000:
            # Shapiro-Wilk test
            w_stat, shapiro_p = shapiro(column_data)
            return w_stat, shapiro_p
            
        # For larger samples, only use D'Agostino-Pearson
        else:
            k2_stat, dagostino_p = normaltest(column_data)
            return k2_stat, dagostino_p
            
    def comparison_of_means(self, data_filename: str, x: str, y: str = None, test_type: str = None, popmean: float = None):
        """
        Performs appropriate statistical test for comparing means based on data characteristics.
        
        Parameters:
        - data_filename: str, path to data file
        - x: str, name of first variable/group
        - y: str, optional, name of second variable/group 
        - test_type: str, optional, force specific test type ('independent', 'paired', 'one-sample')
        - popmean: float, optional, population mean for one-sample test
        
        Returns:
        - test_statistic: float, the test statistic (t or F)
        - p_value: float, the p-value
        - test_used: str, name of statistical test that was used
        """
        # Read the data
        data: pd.DataFrame = self._read_dataframe(data_filename)
        
        # Get data for x
        x_data = data[x]
        
        # If no y provided, do one-sample t-test
        if y is None:
            if popmean is None:
                raise ValueError("Must provide popmean for one-sample t-test")
            stat, p = ttest_1samp(x_data, popmean)
            return stat, p, "one-sample t-test"
            
        # Get data for y
        y_data = data[y]
        
        # If test type explicitly specified, use that
        if test_type:
            if test_type == 'independent':
                stat, p = ttest_ind(x_data, y_data)
                return stat, p, "independent t-test"
            elif test_type == 'paired':
                if len(x_data) != len(y_data):
                    raise ValueError("Groups must be same size for paired test")
                stat, p = ttest_rel(x_data, y_data) 
                return stat, p, "paired t-test"
                
        # Otherwise determine appropriate test
        # If data appears paired (same length and high correlation)
        if len(x_data) == len(y_data):
            correlation = pearsonr(x_data, y_data)[0]
            if abs(correlation) > 0.5:  # Strong correlation suggests paired data
                stat, p = ttest_rel(x_data, y_data)
                return stat, p, "paired t-test"
        
        # Default to independent t-test
        stat, p = ttest_ind(x_data, y_data)
        return stat, p, "independent t-test"
    
    def perform_non_parametric_test(self, data_filename: str, x: str, y: str) -> tuple[float, float, str]:
        """
        Performs appropriate non-parametric test between two variables based on data characteristics.
        
        Args:
        - data_filename: str, path to data file
        - x: str, name of first variable/group
        - y: str, name of second variable/group
        
        Returns:
        - test_statistic: float, the test statistic
        - p_value: float, the p-value 
        - test_used: str, name of statistical test that was used
        """
        # Read the data
        data: pd.DataFrame = self._read_dataframe(data_filename)
        
        x_data = data[x]
        y_data = data[y]
        
        # Check for normality using Shapiro-Wilk test
        _, x_norm_p = shapiro(x_data)
        _, y_norm_p = shapiro(y_data)
        
        # If data is paired (same length and correlated)
        if len(x_data) == len(y_data):
            correlation = pearsonr(x_data, y_data)[0]
            if abs(correlation) > 0.5:  # Strong correlation suggests paired data
                # Use Wilcoxon signed-rank test for paired non-normal data
                stat, p = wilcoxon(x_data, y_data)
                return stat, p, "Wilcoxon signed-rank test"
        
        # For independent samples:
        # If both samples are normally distributed (p > 0.05), t-test would be better
        # But since this is specifically for non-parametric tests:
        
        # Use Mann-Whitney U test for independent samples
        # Especially good when:
        # 1. Data is ordinal
        # 2. Sample sizes are small
        # 3. Data is not normally distributed
        stat, p = mannwhitneyu(x_data, y_data, alternative='two-sided')
        return stat, p, "Mann-Whitney U test"

    def chi_square_test(self, data_filename: str, x: str, y: str) -> tuple[float, float, str]:
        """
        Performs appropriate categorical variable test between two variables based on data characteristics.
        
        Args:
        - data_filename: str, path to data file
        - x: str, name of first categorical variable
        - y: str, name of second categorical variable
        
        Returns:
        - test_statistic: float, the test statistic
        - p_value: float, the p-value
        - test_used: str, name of statistical test that was used
        """
        # Read the data
        data: pd.DataFrame = self._read_dataframe(data_filename)
        
        # Create contingency table
        contingency_table = pd.crosstab(data[x], data[y])
        
        # Check if sample size is small (any expected frequency < 5)
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        if (expected < 5).any():
            # Use Fisher's exact test for small samples
            odds_ratio, p = fisher_exact(contingency_table)
            return odds_ratio, p, "Fisher's exact test"
            
        # For larger samples, use chi-square test of independence
        return chi2, p, "Chi-square test of independence"

    def linear_regression_analysis(self, data_filename: str, x: str, y: str) -> tuple[dict, dict]:
        """
        Performs linear regression analysis between two variables and returns detailed statistics.
        
        Args:
            data_filename: str, path to data file
            x: str, name of independent variable column
            y: str, name of dependent variable column
            
        Returns:
            tuple containing:
            - model_stats: dict with regression statistics (R², adjusted R², F-stat, p-value)
            - coefficients: dict with coefficient statistics (slope, intercept, std errors, p-values)
        """
        # Read the data
        data: pd.DataFrame = self._read_dataframe(data_filename)
        
        # Perform linear regression using scipy for detailed statistics
        slope, intercept, r_value, p_value, std_err = linregress(data[x], data[y])
        
        # Calculate predictions and residuals
        y_pred = slope * data[x] + intercept
        residuals = data[y] - y_pred
        
        # Calculate additional statistics
        n = len(data)
        k = 1  # number of predictors
        r_squared = r_value ** 2
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
        
        # Calculate F-statistic and its p-value
        f_stat = r_squared * (n - 2) / (1 - r_squared)
        f_p_value = 1 - stats.f.cdf(f_stat, 1, n - 2)
        
        # Organize results
        model_stats = {
            'r_squared': r_squared,
            'adj_r_squared': adj_r_squared,
            'f_statistic': f_stat,
            'f_p_value': f_p_value,
            'sample_size': n
        }
        
        coefficients = {
            'slope': slope,
            'intercept': intercept,
            'std_error': std_err,
            'p_value': p_value
        }
        
        return model_stats, coefficients

    def multiple_regression_analysis(self, data_filename: str, x: list[str], y: str) -> tuple[dict, dict]:
        """
        Performs multiple regression analysis between multiple independent variables and one dependent variable.
        
        Args:
            data_filename: str, path to data file
            x: list[str], names of independent variable columns
            y: str, name of dependent variable column
            
        Returns:
            tuple containing:
            - model_stats: dict with regression statistics (R², adjusted R², F-stat, p-value)
            - coefficients: dict with coefficient statistics (coefficients, std errors, p-values)
        """
        # Read the data
        data: pd.DataFrame = self._read_dataframe(data_filename)
        
        # Fit the model
        X = data[x]
        Y = data[y]
        model = LinearRegression()
        model.fit(X, Y)
        
        # Calculate predictions and residuals
        y_pred = model.predict(X)
        residuals = Y - y_pred
        
        # Calculate statistics
        n = len(data)
        k = len(x)  # number of predictors
        r_squared = model.score(X, Y)
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
        
        # Calculate F-statistic and p-value
        mse_model = np.sum((y_pred - Y.mean())**2) / k
        mse_resid = np.sum(residuals**2) / (n - k - 1)
        f_stat = mse_model / mse_resid
        f_p_value = 1 - stats.f.cdf(f_stat, k, n - k - 1)
        
        # Calculate coefficient standard errors and p-values
        mse = np.sum(residuals**2) / (n - k - 1)
        X_with_const = np.column_stack([np.ones(n), X])
        var_covar_matrix = mse * np.linalg.inv(X_with_const.T.dot(X_with_const))
        std_errors = np.sqrt(np.diag(var_covar_matrix))
        
        # Calculate coefficient p-values
        t_stats = np.r_[model.intercept_, model.coef_] / std_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))
        
        # Organize results
        model_stats = {
            'r_squared': r_squared,
            'adj_r_squared': adj_r_squared,
            'f_statistic': f_stat,
            'f_p_value': f_p_value,
            'sample_size': n
        }
        
        coefficients = {
            'features': ['intercept'] + x,
            'coefficients': np.r_[model.intercept_, model.coef_],
            'std_errors': std_errors,
            'p_values': p_values
        }
        
        return model_stats, coefficients

    def effect_size_calculation(self, data_filename: str, x: str, y: str):
        """
        Calculates the effect size between two variables.
        """

        # read the data
        data: pd.DataFrame = self._read_dataframe(data_filename)

        # calculate the effect size
        effect_size = self.cohens_d(data[x], data[y])

        return effect_size
    
    def distribution_fit_test(self, data_filename: str, x: str):
        """
        Tests if data follows a normal distribution
        
        Returns:
        - distribution parameters
        - test results
        - whether data is normally distributed (at 0.05 significance)
        """

        # read the data
        data: pd.DataFrame = self._read_dataframe(data_filename)

        # Fit normal distribution
        mean, std = stats.norm.fit(data[x])
        
        # Test goodness of fit
        kstest_stat, p_value = stats.kstest(data[x], 'norm')
        
        # Interpret results
        is_normal = p_value > 0.05
        
        return {
            'parameters': {'mean': mean, 'std': std},
            'test_statistic': kstest_stat,
            'p_value': p_value,
            'is_normal': is_normal
        }

    def analyze_power_from_data(self, dataframe: pd.DataFrame, x: str, y: str, alpha: float = 0.05):
        """
        Calculates actual effect size from data and determines statistical power
        
        Parameters:
        - data1: first group's data
        - data2: second group's data
        - alpha: significance level (default 0.05)
        
        Returns:
        - dict with effect size, power, and sample size requirements
        """
        # Calculate Cohen's d effect size from actual data
        n1, n2 = len(dataframe[x]), len(dataframe[y])
        var1, var2 = np.var(dataframe[x], ddof=1), np.var(dataframe[y], ddof=1)
        
        # Pooled standard deviation
        pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        # Effect size (Cohen's d)
        effect_size = abs(np.mean(dataframe[x]) - np.mean(dataframe[y])) / pooled_sd
        
        # Calculate power based on actual effect size
        analysis = TTestPower()
        observed_power = analysis.solve_power(
            effect_size=effect_size,
            nobs=min(n1, n2),  # Use smaller sample size for conservative estimate
            alpha=alpha
        )
        
        # If power is low, calculate required sample size
        if observed_power < 0.8:
            required_n = analysis.solve_power(
                effect_size=effect_size,
                power=0.8,
                alpha=alpha
            )
        else:
            required_n = min(n1, n2)
        
        return {
            'effect_size': effect_size,
            'observed_power': observed_power,
            'current_sample_size': min(n1, n2),
            'required_sample_size': required_n,
            'is_power_sufficient': observed_power >= 0.8
        }

    def power_analysis(self, data_filename: str, group1_col: str, group2_col: str):
        """
        Performs power analysis on two groups from your data and provides interpretation
        
        Args:
            data_filename: str, path to data file
            group1_col: str, name of first group column
            group2_col: str, name of second group column
            
        Returns:
            dict containing:
            - results: raw power analysis results
            - interpretation: text interpretation of the results
        """
        data: pd.DataFrame = self._read_dataframe(data_filename)
        
        results = self.analyze_power_from_data(
            data,
            group1_col, 
            group2_col
        )
        
        # Interpret effect size based on Cohen's conventions
        effect_size = results['effect_size']
        if effect_size < 0.2:
            effect_magnitude = "very small"
        elif effect_size < 0.5:
            effect_magnitude = "small"
        elif effect_size < 0.8:
            effect_magnitude = "medium"
        else:
            effect_magnitude = "large"
            
        # Build interpretation
        if results['is_power_sufficient']:
            interpretation = (
                f"Your study has adequate statistical power ({results['observed_power']:.2f}) "
                f"to detect a {effect_magnitude} effect size of {results['effect_size']:.2f}. "
                f"The current sample size of {results['current_sample_size']} per group is sufficient."
            )
        else:
            interpretation = (
                f"Your study has low statistical power ({results['observed_power']:.2f}) "
                f"to detect the {effect_magnitude} effect size of {results['effect_size']:.2f}. "
                f"The current sample size is {results['current_sample_size']} per group, but "
                f"you would need approximately {int(results['required_sample_size'])} samples per group "
                f"to achieve adequate power (0.8)."
            )
        
        return {
            'results': results,
            'interpretation': interpretation
        }

    def perform_anova_analysis(self, 
        data: pd.DataFrame,
        dependent_var: str,
        group_var: str,
        alpha: float = 0.05
    ) -> Dict[str, Union[float, str, Dict]]:
        """
        Performs a comprehensive one-way ANOVA analysis with validation and assumptions checking.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The input dataframe containing the data
        dependent_var : str
            The name of the dependent (continuous) variable
        group_var : str
            The name of the grouping (categorical) variable
        alpha : float
            Significance level (default 0.05)
            
        Returns:
        --------
        Dict containing:
            - test_results: ANOVA test statistics
            - assumptions: Results of assumption checks
            - descriptive_stats: Summary statistics by group
            - post_hoc: Post-hoc test results if ANOVA is significant
            - recommendations: Suggested actions based on results
        """
        try:
            # 1. Input Validation
            if not isinstance(data, pd.DataFrame):
                raise TypeError("Data must be a pandas DataFrame")
                
            if dependent_var not in data.columns or group_var not in data.columns:
                raise ValueError("Specified variables not found in dataframe")
                
            # Check for minimum number of groups
            groups = data[group_var].unique()
            if len(groups) < 2:
                raise ValueError("ANOVA requires at least 2 groups")
                
            # 2. Data Cleaning
            # Remove missing values
            clean_data = data[[dependent_var, group_var]].dropna()
            if len(clean_data) < len(data):
                logging.warning(f"Removed {len(data) - len(clean_data)} rows with missing values")
                
            # 3. Check Sample Sizes
            group_sizes = clean_data.groupby(group_var).size()
            min_group_size = 30  # recommended minimum for normality assumption
            if any(group_sizes < min_group_size):
                logging.warning(f"Some groups have less than {min_group_size} samples")
                
            # 4. Descriptive Statistics
            descriptive_stats = clean_data.groupby(group_var)[dependent_var].agg([
                'count', 'mean', 'std', 'min', 'max'
            ])
            
            # 5. Check Assumptions
            assumptions = self.check_anova_assumptions(clean_data, dependent_var, group_var)
            
            # 6. Perform ANOVA
            groups_data = [group[dependent_var].values for name, group in clean_data.groupby(group_var)]
            f_statistic, p_value = stats.f_oneway(*groups_data)
            
            # 7. Effect Size (Eta-squared)
            ss_between = sum(len(group) * (np.mean(group) - np.mean(clean_data[dependent_var]))**2 
                            for group in groups_data)
            ss_total = sum((clean_data[dependent_var] - np.mean(clean_data[dependent_var]))**2)
            eta_squared = ss_between / ss_total
            
            # 8. Post-hoc Analysis (if ANOVA is significant)
            post_hoc_results = None
            if p_value < alpha:
                post_hoc_results = self.perform_post_hoc_analysis(clean_data, dependent_var, group_var)
                
            # 9. Generate Recommendations
            recommendations = self.generate_recommendations(
                assumptions, 
                p_value, 
                alpha, 
                eta_squared, 
                group_sizes
            )
            
            return {
                'test_results': {
                    'f_statistic': f_statistic,
                    'p_value': p_value,
                    'eta_squared': eta_squared,
                    'significant': p_value < alpha
                },
                'assumptions': assumptions,
                'descriptive_stats': descriptive_stats.to_dict(),
                'post_hoc': post_hoc_results,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logging.error(f"Error in ANOVA analysis: {str(e)}")
            raise

    def check_anova_assumptions(self, 
        data: pd.DataFrame, 
        dependent_var: str, 
        group_var: str
    ) -> Dict[str, Dict[str, Union[bool, float, str]]]:
        """
        Checks the three main assumptions for ANOVA:
        1. Normality
        2. Homogeneity of variance
        3. Independence (note: can only check for obvious violations)
        """
        assumptions = {}
        
        # 1. Normality Test (Shapiro-Wilk for each group)
        normality_results = {}
        for group in data[group_var].unique():
            group_data = data[data[group_var] == group][dependent_var]
            if len(group_data) < 3:
                stat, p_value = np.nan, np.nan
            else:
                stat, p_value = stats.shapiro(group_data)
            normality_results[group] = {
                'statistic': stat,
                'p_value': p_value,
                'normal': p_value > 0.05 if not np.isnan(p_value) else None
            }
        assumptions['normality'] = normality_results
        
        # 2. Homogeneity of Variance (Levene's test)
        groups_data = [group[dependent_var].values 
                    for name, group in data.groupby(group_var)]
        levene_stat, levene_p = stats.levene(*groups_data)
        assumptions['homogeneity'] = {
            'statistic': levene_stat,
            'p_value': levene_p,
            'equal_variance': levene_p > 0.05
        }
        
        # 3. Check for obvious independence violations
        # Look for temporal autocorrelation if data has time component
        if 'time' in data.columns or 'date' in data.columns:
            time_col = 'time' if 'time' in data.columns else 'date'
            autocorr = data.groupby(group_var)[dependent_var].apply(
                lambda x: x.autocorr() if len(x) > 1 else np.nan
            )
            has_autocorr = any(abs(autocorr) > 0.7)  # threshold for strong autocorrelation
            assumptions['independence'] = {
                'temporal_autocorrelation': has_autocorr,
                'warning': "Possible temporal dependence detected" if has_autocorr else "No obvious violations"
            }
        else:
            assumptions['independence'] = {
                'warning': "Independence assumption cannot be tested directly"
            }
        
        return assumptions

    def perform_post_hoc_analysis(self, 
        data: pd.DataFrame, 
        dependent_var: str, 
        group_var: str
    ) -> Dict[str, Union[float, bool]]:
        """
        Performs Tukey's HSD test for post-hoc analysis
        """
        
        # Perform Tukey's test
        tukey = pairwise_tukeyhsd(
            data[dependent_var],
            data[group_var]
        )
        
        # Convert results to dictionary
        results = {
            f"{pair[0]}_vs_{pair[1]}": {
                'mean_diff': pair[2],
                'p_value': pair[3],
                'significant': pair[3] < 0.05,
                'conf_lower': pair[4],
                'conf_upper': pair[5]
            }
            for pair in tukey.summary().data[1:]  # Skip header row
        }
        
        return results

    def generate_recommendations(self, 
        assumptions: Dict,
        p_value: float,
        alpha: float,
        effect_size: float,
        group_sizes: pd.Series
    ) -> List[str]:
        """
        Generates recommendations based on test results and assumption checks
        """
        recommendations = []
        
        # Sample size recommendations
        if any(group_sizes < 30):
            recommendations.append(
                "Consider collecting more data for groups with less than 30 samples"
            )
        
        # Normality violations
        normality_violated = any(
            group['p_value'] < 0.05 
            for group in assumptions['normality'].values() 
            if not np.isnan(group['p_value'])
        )
        if normality_violated:
            recommendations.append(
                "Consider using non-parametric Kruskal-Wallis test due to normality violations"
            )
        
        # Homogeneity of variance violations
        if not assumptions['homogeneity']['equal_variance']:
            recommendations.append(
                "Consider using Welch's ANOVA due to unequal variances"
            )
        
        # Effect size interpretation
        if p_value < alpha:
            if effect_size < 0.01:
                recommendations.append("Effect size is very small")
            elif effect_size < 0.06:
                recommendations.append("Effect size is small")
            elif effect_size < 0.14:
                recommendations.append("Effect size is medium")
            else:
                recommendations.append("Effect size is large")
        
        # Independence concerns
        if 'temporal_autocorrelation' in assumptions['independence']:
            if assumptions['independence']['temporal_autocorrelation']:
                recommendations.append(
                    "Consider using repeated measures ANOVA or mixed-effects model"
                )
        
        return recommendations

    def detect_outliers_iqr(self, data: pd.DataFrame, x: str, threshold: float = 1.5):
        """
        Basic IQR method for outlier detection.
        Good for:
        - Simple, univariate data
        - Normal or near-normal distributions
        - Quick initial analysis
        """
        Q1 = data[x].quantile(0.25)
        Q3 = data[x].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return data[[x]][(data[x] < lower_bound) | (data[x] > upper_bound)]

    def detect_outliers_zscore(self, data: pd.DataFrame, x: str, threshold: float = 3):
        """
        Z-score method for outlier detection.
        Good for:
        - Normally distributed data
        - When you want to consider standard deviations
        """
        z_scores = (data[x] - data[x].mean()) / data[x].std()
        return data[[x]][abs(z_scores) > threshold]

    def detect_outliers_advanced(self, data: pd.DataFrame, method='isolation_forest'):
        """
        Advanced outlier detection methods.
        Use when:
        - Dealing with multiple variables (multivariate)
        - Complex patterns or relationships
        - Non-normal distributions
        - Large datasets
        """
        if method == 'isolation_forest':
            clf = IsolationForest(contamination=0.1, random_state=42)
        elif method == 'local_outlier_factor':
            clf = LocalOutlierFactor(contamination=0.1)
        elif method == 'robust_covariance':
            clf = EllipticEnvelope(contamination=0.1, random_state=42)
            
        outliers = clf.fit_predict(data)
        return data[outliers == -1]

    def should_use_simple_methods(self, data: pd.DataFrame, column: str) -> bool:
        """
        Determines if simple outlier detection methods are appropriate
        """
        # Check sample size
        if len(data) < 1000:
            return True
            
        # Check for normality
        _, normality_p = stats.normaltest(data[column])
        if normality_p > 0.05:
            return True
            
        # Check dimensionality
        if data.shape[1] == 1:
            return True
            
        return False
    
    def should_use_advanced_methods(self, data: pd.DataFrame) -> bool:
        """
        Determines if advanced outlier detection methods are needed
        """
        # Check for multivariate relationships
        if data.shape[1] > 1:
            return True
            
        # Check for large datasets
        if len(data) > 10000:
            return True
            
        # Check for complex distributions
        for column in data.columns:
            if self.is_complex_distribution(data[column]):
                return True
                
        return False

    def is_complex_distribution(self, data: pd.DataFrame, x: str) -> bool:
        """
        Checks if data has complex distribution
        """
        # Extract series from dataframe
        series = data[x]
        
        # Check for multimodality
        kde = gaussian_kde(series)
        x_vals = np.linspace(series.min(), series.max(), 100)
        y = kde(x_vals)
        peaks = len([i for i in range(1, len(y)-1) if y[i-1] < y[i] > y[i+1]])
        
        # Check for skewness and kurtosis
        skew = stats.skew(series)
        kurt = stats.kurtosis(series)
        
        return peaks > 1 or abs(skew) > 2 or abs(kurt) > 7
    
    def detect_outliers(self, data_filename: str, column: str = None):
        """
        Automatically selects and applies appropriate outlier detection method
        """

        # Read the data without cleaning for outliers
        data: pd.DataFrame = self._read_dataframe(data_filename, outlier_method=None)
        
        if column is not None:
            data = data[[column]]
        
        if self.should_use_simple_methods(data, column):
            return self.detect_outliers_iqr(data[column])
        elif self.should_use_advanced_methods(data):
            return self.detect_outliers_advanced(data)
        else:
            # Use combination of methods
            iqr_outliers = self.detect_outliers_iqr(data[column])
            zscore_outliers = self.detect_outliers_zscore(data[column])
            advanced_outliers = self.detect_outliers_advanced(data)
            
            # Combine results (conservative approach)
            combined_outliers = pd.concat([
                iqr_outliers,
                zscore_outliers,
                advanced_outliers
            ]).drop_duplicates()
            
            return combined_outliers

    def text_similarity_raw_text(self, text1: str, text2: str) -> float:
        """
        Calculates text similarity using multiple techniques:
        - TF-IDF cosine similarity
        - Linguistic similarity (spaCy)
        - Sequence matching
        - Entity-based similarity
        
        Parameters:
        -----------
        text1: str
            First text string
        text2: str
            Second text string
            
        Returns:
        --------
        float
            Weighted similarity score between 0 and 1
        """
        
        # Preprocess texts
        clean_text1 = self.preprocess_text(text1)
        clean_text2 = self.preprocess_text(text2)
        
        # 1. TF-IDF Cosine Similarity
        tfidf_matrix = self.tfidf.fit_transform([clean_text1, clean_text2])
        tfidf_sim = (tfidf_matrix * tfidf_matrix.T).A[0,1]
        
        # 2. Linguistic Similarity (spaCy)
        doc1 = self.nlp(clean_text1)
        doc2 = self.nlp(clean_text2)
        linguistic_sim = doc1.similarity(doc2)
        
        # 3. Sequence Similarity
        sequence_sim = SequenceMatcher(None, clean_text1, clean_text2).ratio()

        # 4. Entity Similarity
        entities1 = set([ent.text.lower() for ent in self.nlp(text1).ents])
        entities2 = set([ent.text.lower() for ent in self.nlp(text2).ents])
        entity_sim = len(entities1.intersection(entities2)) / max(
            len(entities1.union(entities2)), 1
        ) if entities1 or entities2 else 0.0
        
        # Calculate weighted average
        weights = {
            'tfidf': 0.35,
            'linguistic': 0.35,
            'sequence': 0.15,
            'entity': 0.15
        }
        
        weighted_score = (
            tfidf_sim * weights['tfidf'] +
            linguistic_sim * weights['linguistic'] +
            sequence_sim * weights['sequence'] +
            entity_sim * weights['entity']
        )
        
        return weighted_score
    
    def text_similarity_from_dataframe(self, data_filename: str, column1: str, column2: str) -> float:
        """
        Calculates text similarity between two columns in a dataframe using spaCy similarity
        
        Parameters
        ----------
        data_filename: str
            Path to CSV file containing the data
        column1: str 
            Name of first text column
        column2: str
            Name of second text column
            
        Returns
        -------
        float
            Average spaCy similarity score between the columns
        """

        # Read the data
        data: pd.DataFrame = self._read_dataframe(data_filename)
        
        # Convert columns to spaCy docs
        docs1 = [self.nlp(str(text)) for text in data[column1]]
        docs2 = [self.nlp(str(text)) for text in data[column2]]
        
        # Calculate similarity for each row and take mean
        similarities = [doc1.similarity(doc2) for doc1, doc2 in zip(docs1, docs2)]
        return np.mean(similarities)

    def sentiment_analysis(self, text: str) -> Dict[str, float]:
        """
        Performs sentiment analysis on a text string to detect emotional tone and polarity.
        
        Parameters
        ----------
        text : str
            The input text to analyze
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing sentiment scores:
            - polarity: float between -1 (negative) and 1 (positive)
            - subjectivity: float between 0 (objective) and 1 (subjective)
            - compound: float between -1 (most negative) and 1 (most positive)
        """
        # Preprocess the text
        text = self.preprocess_text(text)
        
        # Get spaCy doc
        doc = self.nlp(text)
        
        # Calculate polarity score (-1 to 1)
        polarity = sum([token.sentiment for token in doc]) / len(doc)
        
        # Calculate subjectivity (0 to 1)
        subjectivity = len([token for token in doc if token.pos_ in ['ADJ', 'ADV']]) / len(doc)
        
        # Calculate compound score using token weights
        weights = {'ADJ': 2.0, 'VERB': 1.5, 'ADV': 1.0, 'NOUN': 0.5}
        weighted_scores = []
        for token in doc:
            score = token.sentiment * weights.get(token.pos_, 0.5)
            weighted_scores.append(score)
        compound = sum(weighted_scores) / len(weighted_scores)
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity, 
            'compound': max(min(compound, 1.0), -1.0)
        }

    def seasonality_test(self, data_filename: str, column: str, period: int = 12) -> Dict[str, float]:
        """
        Performs comprehensive seasonal analysis on time series data using multiple methods
        to ensure robust detection of seasonality patterns.
        
        Parameters:
        -----------
        data_filename : str
            Path to the data file
        column : str 
            Name of the column containing the time series data
        period : int, default=12
            Expected seasonality period (e.g., 12 for monthly data)
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing:
            - seasonal_strength: float (0-1 scale)
            - seasonal_peaks: List[int] (indices of detected peaks)
            - acf_seasonal: float (autocorrelation at seasonal lag)
            - kw_test_pvalue: float (Kruskal-Wallis test p-value)
            - recommendations: List[str] (analysis insights)
        """
        
        # Read and prepare data
        data = self._read_dataframe(data_filename, outlier_method=None)
        series = data[column].astype(float)
        
        # Handle missing values
        series = series.interpolate(method='linear')
        
        # Ensure minimum length for analysis
        if len(series) < period * 2:
            return {
                "error": "Time series too short for seasonal analysis",
                "min_required_length": period * 2,
                "current_length": len(series)
            }
        
        results = {}
        recommendations = []
        
        try:
            # 1. Decomposition-based Analysis
            decomposition = seasonal_decompose(
                series,
                period=period,
                extrapolate_trend=True
            )
            
            # Calculate seasonal strength
            seasonal_var = np.var(decomposition.seasonal)
            residual_var = np.var(decomposition.resid)
            seasonal_strength = seasonal_var / (seasonal_var + residual_var)
            results['seasonal_strength'] = float(seasonal_strength)
            
            # 2. Autocorrelation Analysis
            acf_values = acf(series, nlags=period*2)
            seasonal_acf = acf_values[period]
            results['acf_seasonal'] = float(seasonal_acf)
            
            # 3. Kruskal-Wallis Test for Seasonality
            seasonal_groups = [series[i::period] for i in range(period)]
            h_stat, kw_pvalue = stats.kruskal(*seasonal_groups)
            results['kw_test_pvalue'] = float(kw_pvalue)
            
            # 4. Peak Detection
            peaks, _ = find_peaks(decomposition.seasonal, distance=period//2)
            results['seasonal_peaks'] = peaks.tolist()
            
            # 5. Analysis and Recommendations
            if seasonal_strength > 0.6:
                recommendations.append("Strong seasonal pattern detected")
            elif seasonal_strength > 0.3:
                recommendations.append("Moderate seasonal pattern detected")
            else:
                recommendations.append("Weak or no seasonal pattern detected")
                
            if abs(seasonal_acf) > 0.7:
                recommendations.append("Strong seasonal autocorrelation confirmed")
            
            if kw_pvalue < 0.05:
                recommendations.append("Statistically significant seasonal differences found")
            
            # Handle edge cases
            if np.std(series) < 1e-6:
                recommendations.append("WARNING: Near-constant series detected")
                
            if len(peaks) < 2:
                recommendations.append("WARNING: Insufficient seasonal peaks detected")
                
            # Check for trend interference
            trend_strength = np.var(decomposition.trend) / np.var(series)
            if trend_strength > 0.7:
                recommendations.append("Strong trend may be masking seasonal patterns")
                
            # Check for data quality
            missing_pct = data[column].isnull().mean() * 100
            if missing_pct > 5:
                recommendations.append(f"WARNING: {missing_pct:.1f}% missing values may affect reliability")
            
            results['recommendations'] = recommendations
            
        except Exception as e:
            return {
                "error": f"Analysis failed: {str(e)}",
                "recommendations": ["Please check data quality and format"]
            }
        
        return results

    def stationarity_test(self, data_filename: str, column: str) -> Dict[str, Any]:
        """
        Performs comprehensive stationarity analysis using multiple statistical tests
        and handles various edge cases based on empirical research.
        
        Parameters:
        -----------
        data_filename : str
            Path to the data file
        column : str
            Name of the column containing the time series data
            
        Returns:
        --------
        Dict containing:
            - adf_test: Dict with ADF test results
            - kpss_test: Dict with KPSS test results
            - rolling_stats: Dict with rolling statistics
            - transformation_suggestions: List of recommended transformations
            - warnings: List of potential issues detected
        """
        
        # Read and prepare data
        data = self._read_dataframe(data_filename)
        series = data[column].astype(float)
        
        results = {}
        warnings = []
        
        # Handle missing values
        missing_pct = series.isnull().mean() * 100
        if missing_pct > 0:
            warnings.append(f"Series contains {missing_pct:.1f}% missing values")
            # Interpolate missing values for analysis
            series = series.interpolate(method='linear')
        
        # Check minimum length requirement
        if len(series) < 30:
            return {
                "error": "Series too short for reliable stationarity testing",
                "min_required_length": 30,
                "current_length": len(series)
            }
        
        try:
            # 1. Augmented Dickey-Fuller Test
            adf_result = adfuller(series, autolag='AIC')
            results['adf_test'] = {
                'statistic': float(adf_result[0]),
                'pvalue': float(adf_result[1]),
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < 0.05
            }
            
            # 2. KPSS Test
            kpss_result = kpss(series, regression='c', nlags="auto")
            results['kpss_test'] = {
                'statistic': float(kpss_result[0]),
                'pvalue': float(kpss_result[1]),
                'critical_values': kpss_result[3],
                'is_stationary': kpss_result[1] > 0.05
            }
            
            # 3. Rolling Statistics Analysis
            window = min(len(series) // 4, 252)  # Use quarter of data or 1 year of daily data
            rolling_mean = series.rolling(window=window).mean()
            rolling_std = series.rolling(window=window).std()
            
            # Check for trend using rolling statistics
            mean_trend = np.abs(rolling_mean.iloc[-1] - rolling_mean.iloc[window]) / rolling_mean.std()
            results['rolling_stats'] = {
                'mean_trend_strength': float(mean_trend),
                'std_stability': float(rolling_std.std() / rolling_std.mean())
            }
            
            # 4. Additional Checks and Edge Cases
            
            # Check for variance stability
            if rolling_std.std() / rolling_std.mean() > 0.5:
                warnings.append("Heteroscedasticity detected - consider variance stabilizing transformation")
            
            # Check for outliers using IQR method
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            outliers_pct = ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))).mean() * 100
            if outliers_pct > 5:
                warnings.append(f"High percentage of outliers detected ({outliers_pct:.1f}%)")
            
            # Check for seasonality
            if len(series) >= 2 * 12:  # Need at least 2 years for seasonal check
                seasonal_decompose = stats.pearsonr(
                    series[:-12],
                    series[12:]
                )[0]
                results['seasonality_correlation'] = float(seasonal_decompose)
                
                if abs(seasonal_decompose) > 0.7:
                    warnings.append("Strong seasonal pattern detected - consider seasonal differencing")
            
            # 5. Transformation Suggestions
            transformations = []
            
            # Log transformation for variance stabilization
            if series.std() / series.mean() > 0.3:
                transformations.append("Log transformation")
            
            # Differencing for trend removal
            if not results['adf_test']['is_stationary']:
                transformations.append("First differencing")
            
            # Seasonal differencing if seasonal pattern detected
            if 'seasonality_correlation' in results and abs(results['seasonality_correlation']) > 0.7:
                transformations.append("Seasonal differencing")
            
            results['transformation_suggestions'] = transformations
            results['warnings'] = warnings
            
            # 6. Overall Stationarity Assessment
            results['is_stationary'] = (
                results['adf_test']['is_stationary'] and 
                results['kpss_test']['is_stationary'] and
                len(warnings) == 0
            )
            
        except Exception as e:
            return {
                "error": f"Analysis failed: {str(e)}",
                "warnings": warnings
            }
        
        return results

    def clustering_analysis(self, data_filename: str, column: str, max_clusters: int = 10) -> Dict[str, Any]:
        """
        Performs comprehensive clustering analysis using multiple algorithms and handles edge cases.
        Uses ensemble approach to determine optimal clustering configuration.
        
        Parameters:
        -----------
        data_filename : str
            Path to the data file
        column : str
            Name of the column to analyze
        max_clusters : int, default=10
            Maximum number of clusters to consider
            
        Returns:
        --------
        Dict containing:
            - optimal_clusters: int
            - cluster_labels: array
            - algorithm_scores: Dict
            - silhouette_scores: Dict
            - cluster_metrics: Dict
            - warnings: List[str]
            - recommendations: List[str]
        """
        
        results = {}
        warnings = []
        recommendations = []
        
        try:
            # Read and prepare data
            data = self._read_dataframe(data_filename)
            series = data[column].values.reshape(-1, 1)
            
            # Handle missing values
            missing_pct = np.isnan(series).mean() * 100
            if missing_pct > 0:
                warnings.append(f"Series contains {missing_pct:.1f}% missing values")
                series = self._handle_missing_values(series)
                
            # Check minimum length requirement
            if len(series) < 30:
                return {
                    "error": "Series too short for reliable clustering",
                    "min_required_length": 30,
                    "current_length": len(series)
                }
                
            # Handle outliers and scaling
            scaler = self._choose_scaler(series)
            scaled_data = scaler.fit_transform(series)
            
            # Initialize clustering algorithms with empirically-backed parameters
            clustering_algos = {
                'kmeans': KMeans(random_state=42),
                'minibatch_kmeans': MiniBatchKMeans(random_state=42),
                'agglomerative': AgglomerativeClustering(),
                'dbscan': DBSCAN(eps=0.3, min_samples=5),
                'spectral': SpectralClustering(random_state=42),
                'gaussian_mixture': GaussianMixture(random_state=42),
                'birch': Birch(n_clusters=None),
                'optics': OPTICS(min_samples=5)
            }
            
            # Store results for each algorithm
            algo_results = {}
            silhouette_scores = {}
            calinski_scores = {}
            davies_scores = {}
            
            # Analyze optimal number of clusters
            for algo_name, algo in clustering_algos.items():
                try:
                    if algo_name in ['dbscan', 'optics']:
                        # Density-based clustering
                        labels = algo.fit_predict(scaled_data)
                        n_clusters = len(np.unique(labels[labels >= 0]))
                        
                    elif algo_name in ['kmeans', 'minibatch_kmeans', 'agglomerative', 
                                    'spectral', 'gaussian_mixture', 'birch']:
                        # Try different numbers of clusters
                        best_score = -1
                        best_n = 2
                        
                        for n in range(2, min(max_clusters + 1, len(series) // 5)):
                            if hasattr(algo, 'n_clusters'):
                                algo.n_clusters = n
                            elif hasattr(algo, 'n_components'):
                                algo.n_components = n
                                
                            labels = algo.fit_predict(scaled_data)
                            
                            if len(np.unique(labels)) < 2:
                                continue
                                
                            score = silhouette_score(scaled_data, labels)
                            
                            if score > best_score:
                                best_score = score
                                best_n = n
                                
                        # Use optimal number of clusters
                        if hasattr(algo, 'n_clusters'):
                            algo.n_clusters = best_n
                        elif hasattr(algo, 'n_components'):
                            algo.n_components = best_n
                            
                        labels = algo.fit_predict(scaled_data)
                        
                    # Calculate clustering metrics
                    if len(np.unique(labels)) >= 2:
                        silhouette_scores[algo_name] = silhouette_score(scaled_data, labels)
                        calinski_scores[algo_name] = calinski_harabasz_score(scaled_data, labels)
                        davies_scores[algo_name] = davies_bouldin_score(scaled_data, labels)
                        algo_results[algo_name] = labels
                        
                except Exception as e:
                    warnings.append(f"{algo_name} failed: {str(e)}")
                    continue
                    
            # Select best algorithm based on ensemble of metrics
            if algo_results:
                best_algo = max(silhouette_scores.items(), key=lambda x: x[1])[0]
                optimal_labels = algo_results[best_algo]
                
                results['optimal_clusters'] = len(np.unique(optimal_labels))
                results['cluster_labels'] = optimal_labels.tolist()
                results['algorithm_scores'] = {
                    'silhouette': silhouette_scores,
                    'calinski_harabasz': calinski_scores,
                    'davies_bouldin': davies_scores
                }
                results['best_algorithm'] = best_algo
                
                # Generate recommendations
                if results['optimal_clusters'] == 2:
                    recommendations.append("Binary clustering pattern detected")
                elif results['optimal_clusters'] >= 7:
                    recommendations.append("High number of clusters detected - consider dimensionality reduction")
                    
                # Check cluster balance
                cluster_sizes = np.bincount(optimal_labels)
                size_ratio = np.min(cluster_sizes) / np.max(cluster_sizes)
                if size_ratio < 0.1:
                    warnings.append("Highly imbalanced clusters detected")
                    recommendations.append("Consider using density-based clustering algorithms")
                    
            else:
                return {
                    "error": "No clustering algorithm succeeded",
                    "warnings": warnings
                }
                
            results['warnings'] = warnings
            results['recommendations'] = recommendations
            
        except Exception as e:
            return {
                "error": f"Analysis failed: {str(e)}",
                "warnings": warnings
            }
            
        return results

    def _choose_scaler(self, data: np.ndarray) -> object:
        """
        Chooses appropriate scaler based on data characteristics
        """
        
        # Check for outliers using IQR method
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        outlier_mask = (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))
        outlier_percentage = outlier_mask.mean() * 100
        
        # Use RobustScaler if significant outliers present
        if outlier_percentage > 5:
            return RobustScaler()
        return StandardScaler()

    def _handle_missing_values(self, data: np.ndarray) -> np.ndarray:
        """
        Handles missing values using appropriate interpolation
        """
        
        # Get indices of missing values
        missing_idx = np.isnan(data).flatten()
        
        if missing_idx.all():
            raise ValueError("All values are missing")
        
        # Get indices of non-missing values
        valid_idx = ~missing_idx
        valid_data = data[valid_idx].flatten()
        
        # Perform interpolation
        f = interpolate.interp1d(
            np.where(valid_idx)[0], 
            valid_data,
            kind='cubic',
            fill_value='extrapolate'
        )
        
        return f(np.arange(len(data))).reshape(-1, 1)
        
    def linear_optimization(self, data_filename: str, x_column: str, y_column: str) -> Dict[str, Any]:
        """
        Performs comprehensive linear optimization using multiple methods.
        
        Parameters:
        -----------
        data_filename : str
            Path to the data file
        x_column : str
            Independent variable column name
        y_column : str
            Dependent variable column name
            
        Returns:
        --------
        Dict containing:
            - optimal_parameters: Dict
            - model_performance: Dict
            - optimization_method: str
            - constraints_analysis: Dict
            - recommendations: List[str]
        """
        
        try:
            # Read and prepare data
            data = self._read_dataframe(data_filename)
            X = data[x_column].astype(float).values.reshape(-1, 1)
            y = data[y_column].astype(float).values
            
            results = {}
            warnings = []
            recommendations = []
            
            # Handle missing values and infinities
            mask = np.isfinite(X.flatten()) & np.isfinite(y)
            X = X[mask]
            y = y[mask]
            
            if len(X) < 10:
                return {
                    "error": "Insufficient data for optimization",
                    "min_required": 10,
                    "current_length": len(X)
                }
                
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 1. Try multiple optimization approaches
            models = {
                'linear': LinearRegression(),
                'ridge': Ridge(alpha=1.0),
                'lasso': Lasso(alpha=1.0),
                'elastic_net': ElasticNet(alpha=1.0, l1_ratio=0.5),
                'huber': HuberRegressor(epsilon=1.35),
                'ransac': RANSACRegressor(random_state=42)
            }
            
            model_scores = {}
            model_params = {}
            
            for name, model in models.items():
                try:
                    model.fit(X_scaled, y)
                    y_pred = model.predict(X_scaled)
                    r2 = r2_score(y, y_pred)
                    mse = mean_squared_error(y, y_pred)
                    
                    model_scores[name] = {
                        'r2': r2,
                        'mse': mse,
                        'coefficients': model.coef_.tolist(),
                        'intercept': float(model.intercept_)
                    }
                    
                    model_params[name] = {
                        'coefficients': model.coef_.tolist(),
                        'intercept': float(model.intercept_)
                    }
                    
                except Exception as e:
                    warnings.append(f"{name} optimization failed: {str(e)}")
                    
            # 2. Advanced Optimization using scipy
            def objective(params):
                return np.mean((y - (params[0] * X_scaled.flatten() + params[1])) ** 2)
            
            # Add constraints
            constraints = [
                {'type': 'ineq', 'fun': lambda x: x[0]},  # Positive slope constraint
                {'type': 'ineq', 'fun': lambda x: 1000 - abs(x[1])}  # Bounded intercept
            ]
            
            # Initial guess using best performing model
            best_model = max(model_scores.items(), key=lambda x: x[1]['r2'])[0]
            initial_guess = [
                model_params[best_model]['coefficients'][0],
                model_params[best_model]['intercept']
            ]
            
            # Scipy optimization
            optimization_result = minimize(
                objective,
                initial_guess,
                method='SLSQP',
                constraints=constraints,
                bounds=Bounds([-np.inf, -np.inf], [np.inf, np.inf])
            )
            
            # 3. Analyze results and select best method
            best_r2 = max(score['r2'] for score in model_scores.values())
            best_method = max(model_scores.items(), key=lambda x: x[1]['r2'])[0]
            
            # 4. Check optimization quality
            if best_r2 < 0.5:
                recommendations.append("Poor linear fit - consider non-linear optimization")
            
            if best_r2 > 0.95:
                recommendations.append("Excellent fit achieved - check for overfitting")
            
            # Check residuals for heteroscedasticity
            residuals = y - model_params[best_method]['coefficients'][0] * X_scaled.flatten()
            if np.std(residuals[:len(residuals)//2]) / np.std(residuals[len(residuals)//2:]) > 2:
                recommendations.append("Heteroscedasticity detected - consider weighted optimization")
            
            # 5. Prepare final results
            results = {
                'optimal_parameters': {
                    'scipy_optimization': {
                        'coefficients': optimization_result.x[0],
                        'intercept': optimization_result.x[1],
                        'success': optimization_result.success,
                        'message': optimization_result.message
                    },
                    'best_model': {
                        'method': best_method,
                        'parameters': model_params[best_method]
                    }
                },
                'model_performance': {
                    'best_r2': best_r2,
                    'model_scores': model_scores,
                    'residual_stats': {
                        'mean': float(np.mean(residuals)),
                        'std': float(np.std(residuals))
                    }
                },
                'optimization_method': best_method,
                'constraints_analysis': {
                    'positive_slope': model_params[best_method]['coefficients'][0] > 0,
                    'bounded_intercept': abs(model_params[best_method]['intercept']) < 1000
                },
                'recommendations': recommendations,
                'warnings': warnings
            }
            
        except Exception as e:
            return {
                "error": f"Optimization failed: {str(e)}",
                "warnings": warnings
            }
            
        return results

    def box_plot(self, data_filename: str, x: str, y: str, title: str, color=None, filename="box_plot.html", group_by_binary: bool=False):
        """
        Creates a box plot and saves it as an HTML file.
        
        For numeric x values:
        - Divides into 2 groups based on median
        
        For categorical x values:
        - Groups into top 5 categories by frequency
        - Remaining categories grouped as 'Other'
        
        Parameters:
        - data_filename: str, path to the data file
        - x: str, column name for x-axis data
        - y: str, column name for y-axis data
        - title: str, plot title
        - color: str, optional column for color grouping
        - filename: str, output filename
        - group_by_binary: bool, whether to use binary columns for grouping
        """
        # read the data
        data: pd.DataFrame = self._read_dataframe(data_filename, outlier_method='confidence_interval')
        
        # Create a copy to avoid modifying original data
        plot_data = data.copy()
        
        # Handle x values based on type and cardinality
        if plot_data[x].nunique() > 2:
            if np.issubdtype(plot_data[x].dtype, np.number):
                # For numeric columns, split into two groups based on median
                median = plot_data[x].median()
                plot_data['grouped_x'] = np.where(
                    plot_data[x] <= median,
                    f'≤ {median:.2f}',
                    f'> {median:.2f}'
                )
                # Use the new grouped column for plotting
                x_col = 'grouped_x'
                
            else:
                # For categorical columns, keep top 5 by frequency percentage
                value_counts = plot_data[x].value_counts()
                total_count = len(plot_data)
                
                # Calculate percentages
                percentages = (value_counts / total_count) * 100
                
                # Get top 5 categories
                top_categories = percentages.nlargest(5).index
                
                # Group others
                plot_data['grouped_x'] = plot_data[x].apply(
                    lambda val: val if val in top_categories else 'Other'
                )
                
                # Add percentage to category labels
                category_percentages = plot_data['grouped_x'].value_counts() / len(plot_data) * 100
                plot_data['grouped_x'] = plot_data['grouped_x'].apply(
                    lambda x: f"{x} ({category_percentages[x]:.1f}%)"
                )
                
                # Use the new grouped column for plotting
                x_col = 'grouped_x'
        else:
            # If 2 or fewer categories, use original x column
            x_col = x
        
        # Handle binary grouping if requested
        if group_by_binary:
            binary_columns = self.find_binary_columns(data)
            if binary_columns:
                binary_column = random.choice(binary_columns)
                color = binary_column
        
        # Create the plot
        fig = px.box(plot_data, x=x_col, y=y, color=color, title=title)
        
        # Update layout for better readability
        fig.update_xaxes(
            title=x,
            tickangle=45 if plot_data[x_col].nunique() > 2 else 0
        )
        
        # Add hover data showing original x values if grouped
        if x_col != x:
            fig.update_traces(
                hovertemplate=(
                    f"{x}: %{{customdata}}<br>" +
                    f"{y}: %{{y}}<br>" +
                    "<extra></extra>"
                ),
                customdata=plot_data[x]
            )
        
        fig = self.apply_template(fig)
        self.save_plot_html(fig, filename)
        return fig

       
    
file_path: str = '/content/california_housing_test.csv'
charting_tools = ChartingTools()

charting_tools.density_heatmap_plot(file_path, y='total_rooms', x='population', 
                          title='Total Rooms vs Population')