# Wait for RabbitMQ server to start
sleep 10

# Add RabbitMQ Management User
rabbitmqctl add_user admin 123456
rabbitmqctl set_user_tags admin administrator
rabbitmqctl set_permissions -p / admin ".*" ".*" ".*"


