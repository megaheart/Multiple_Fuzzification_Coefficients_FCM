# Use the official ASP.NET Core runtime as the base image
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS base

# Install Python 3
RUN apt-get update && \
    apt-get install -y python3 python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

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

CMD ["rabbitmq-server"]
