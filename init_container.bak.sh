# #!/bin/bash

# echo "Starting init_container.sh"

# rabbitmq-server &

# sleep 60

# echo "RabbitMQ started"

# cd ./AI
# python3 ./ai_server.py &

# echo "AI server started"

# cd ../Backend
# dotnet ./Backend.dll &

# echo "Backend started"

# # Prevent the script from exiting
# wait