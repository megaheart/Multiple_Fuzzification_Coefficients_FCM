docker build -t datn-img .

docker run --name datn-web -p 15672:15672 -p 5672:5672 -p 8080:80 -p 8081:443 -p 6379:6379 datn-img

docker build -t aspnet-python-rabbitmq-redis:latest .

docker run --name datn-container -p 15672:15672 -p 5672:5672 -p 6379:6379 aspnet-python-rabbitmq-redis:latest

docker build -t node-pnpm:20.14.0 .

docker run -it --entrypoint /bin/sh node-pnpm:20.14.0

heroku container:push web -a datn-linh-tpm