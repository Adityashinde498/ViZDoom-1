# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the AGV Image directory contents into the container at /app
COPY 

# Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# Run a command to confirm the files are there
CMD ["ls", "-l"]