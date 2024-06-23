#!/bin/bash

echo "Starting init_container.sh"

# Start rabbitmq-server and redirect its output to a log file
rabbitmq-server > rabbitmq.log 2>&1 &
rabbitmq_pid=$!

echo "RabbitMQ started"

# Start redis-server and redirect its output to a log file
redis-server /etc/redis/redis.conf > redis.log 2>&1 &
redis_pid=$!

echo "Redis started"

# Use tail to follow the logs of all programs in the background
tail -f rabbitmq.log redis.log &
tail_pid=$!

# Function to clean up background processes on exit
cleanup() {
    kill $rabbitmq_pid $redis_pid $tail_pid
}

# Trap SIGINT and SIGTERM to trigger the cleanup function
trap cleanup SIGINT SIGTERM

# Wait for background processes to finish
wait $rabbitmq_pid $redis_pid
