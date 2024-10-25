# Use the official Python image from the Docker Hub
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy local files into the container
COPY . /app/

# Install make to use it for running the Makefile
RUN apt-get update && apt-get install -y make && rm -rf /var/lib/apt/lists/*

# Run the install target from the Makefile
RUN make install

# Set the entry point for the container
CMD ["python"]