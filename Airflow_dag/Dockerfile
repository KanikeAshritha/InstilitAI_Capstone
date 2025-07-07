# Use the official Apache Airflow image as the base
FROM apache/airflow:2.9.0

# Switch to root user to install system packages
USER root

# Update system packages and install libgomp1 (needed by XGBoost)
RUN apt-get update && apt-get install -y libgomp1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Switch back to airflow user (recommended for running Airflow)
USER airflow

# Copy Python dependencies file into the container
COPY requirements.txt ./

# Install all required Python libraries listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable to ignore all Python syntax warnings globally
ENV PYTHONWARNINGS="ignore::SyntaxWarning"
