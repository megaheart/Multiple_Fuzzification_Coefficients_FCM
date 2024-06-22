docker build -t datn-img .

docker run --name datn-web -p 15672:15672 -p 5672:5672 -p 8080:80 -p 8081:443 datn-img

docker build -t aspnet-python-rabbitmq:latest .

docker run --name datn-container -p 15672:15672 -p 5672:5672 aspnet-python-rabbitmq:latest

docker build -t node-pnpm:20.14.0 .

heroku container:push web -a datn-linh-tpm