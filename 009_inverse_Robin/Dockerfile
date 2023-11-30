# Define base image
FROM continuumio/miniconda3
 
# Set working directory for the project
WORKDIR /app
 
# Create Conda environment from the YAML file
COPY environment.yml .
RUN conda env create -f environment.yml
