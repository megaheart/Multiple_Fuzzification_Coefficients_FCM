#!/bin/bash

echo "Starting init_container.sh"
# Start redis-server and redirect its output to a log file
redis-server /etc/redis/redis.conf > redis.log 2>&1 &
redis_pid=$!

echo "Redis started"

# Start rabbitmq-server and redirect its output to a log file
rabbitmq-server > rabbitmq.log 2>&1 &
rabbitmq_pid=$!

# Wait for RabbitMQ to start
sleep 30

echo "RabbitMQ started"

# Start the backend server and redirect its output to a log file
cd ./Backend
export ASPNETCORE_URLS=http://0.0.0.0:$PORT
export DOTNET_RUNNING_IN_CONTAINER=true
dotnet ./Backend.dll > ../backend.log 2>&1 &
backend_pid=$!
cd ..

echo "Backend started"

# Start the AI server and redirect its output to a log file
cd ./AI
/opt/py3env/bin/python -u ./ai_server.py > ../ai_server.log 2>&1 &
ai_server_pid=$!
cd ..

echo "AI server started"

# Use tail to follow the logs of all programs in the background
tail -f rabbitmq.log ai_server.log backend.log redis.log &
tail_pid=$!

# Function to clean up background processes on exit
cleanup() {
    kill $redis_pid $rabbitmq_pid $ai_server_pid $backend_pid $tail_pid
}

# Trap SIGINT and SIGTERM to trigger the cleanup function
trap cleanup SIGINT SIGTERM

# Wait for background processes to finish
wait $redis_pid $rabbitmq_pid $ai_server_pid $backend_pid
