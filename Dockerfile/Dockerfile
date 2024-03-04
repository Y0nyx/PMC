# Use the TensorFlow GPU image as the base image
FROM pytorch/pytorch:latest

# Install git to clone the repository
RUN apt-get update && apt-get install -y git

# Set up SSH config to prevent "host key verification failed" errors
RUN mkdir -p /root/.ssh && \
    echo "StrictHostKeyChecking no" >> /root/.ssh/config

# Define a build argument for SSH key
ARG SSH_PRIVATE_KEY

# Add your SSH private key to the container
RUN mkdir -p /root/.ssh && \
    echo "$SSH_PRIVATE_KEY" > /root/.ssh/id_rsa && \
    chmod 600 /root/.ssh/id_rsa && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts

# Clone the repository using SSH
RUN git clone -b Pipeline_dockerfile git@github.com:Y0nyx/PMC.git

# Change the working directory to the cloned repository
WORKDIR /workspace/PMC

# Install requirements
RUN pip install -r pipeline_requirements.txt
