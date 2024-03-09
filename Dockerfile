# Use the TensorFlow GPU image as the base image
FROM pytorch/pytorch:latest

# Install git to clone the repository
RUN apt-get update && apt-get install -y git && \
    apt-get install -y libgl1-mesa-dev && \
    apt-get install -y libglib2.0-0
    
# Change the working directory to the cloned repository
WORKDIR /PMC

# Copy the requirements file into the container
COPY unsupervised_docker_requirements.txt .

# Install the Python dependencies
#RUN pip install -r pipeline_requirements.txt

# Copy source code from host machine to Docker image
COPY src/ src/
