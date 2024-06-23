# Use the official ASP.NET Core runtime as the base image
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS base

# Install Python 3
RUN apt-get update && \
    apt-get install -y python3 python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

# Python 3 Virtual Environment Setup
RUN apt-get update && apt-get install -y python3-venv && \
    rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/py3env 
# RUN source /opt/py3env/bin/activate

COPY ./install_rabbitmq.sh /install_rabbitmq.sh

# Install RabbitMQ
RUN chmod +x /install_rabbitmq.sh && \
    /install_rabbitmq.sh

# Expose RabbitMQ ports
EXPOSE 5672 15672

# Set up the RabbitMQ server to start automatically
RUN echo "[rabbitmq_management,rabbitmq_shovel,rabbitmq_shovel_management]." > /etc/rabbitmq/enabled_plugins

# Overwrite configuration file to allow guest user to access RabbitMQ
COPY rabbitmq.conf /etc/rabbitmq/rabbitmq.conf
COPY definitions.json /etc/rabbitmq/definitions.json

# Install Redis
RUN apt-get update && \
    apt-get install -y redis-server && \
    rm -rf /var/lib/apt/lists/*

# Expose Redis port
EXPOSE 6379

# Overwrite configuration file of Redis
COPY ./redis.conf /etc/redis/redis.conf

COPY ./run.sh /run.sh
RUN chmod +x /run.sh

# Start Redis server
CMD ["/run.sh"]
