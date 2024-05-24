using Microsoft.AspNetCore.SignalR;
using RabbitMQ.Client;
using RabbitMQ.Client.Events;
using System.Text;

namespace Backend.Services
{
    public class RabbitMQConsumer:IDisposable, IHostedService
    {
        private readonly string _uri;
        private readonly ConnectionFactory factory;
        private readonly IConnection connection;
        private readonly IModel channel;
        private readonly ILogger<RabbitMQConsumer> _logger;
        private readonly IHubContext<SignalRHub> _hubContext;

        public RabbitMQConsumer(ILogger<RabbitMQConsumer> logger, IConfiguration configuration, IHubContext<SignalRHub> hubContext)
        {
            _uri = configuration.GetConnectionString("RabbitMQ") ?? throw new Exception("rabbitmqUri phải khác null");
            factory = new ConnectionFactory() { Uri = new Uri(_uri) };
            connection = factory.CreateConnection();
            channel = connection.CreateModel();
            _logger = logger;
            _hubContext = hubContext;
        }

        public void Consume(string queue, EventHandler<string> eventHandler)
        {
            
            // Declare queue
            channel.QueueDeclare(queue: queue, durable: false, exclusive: false, autoDelete: false, arguments: null);

            // Create consumer
            var consumer = new EventingBasicConsumer(channel);
            consumer.Received += (model, ea) =>
            {
                var body = ea.Body.ToArray();
                var message = Encoding.UTF8.GetString(body);
                eventHandler(this, message);
                _logger.Log(LogLevel.Information, $"RabbitMQ message: {message}");
                channel.BasicAck(deliveryTag: ea.DeliveryTag, multiple: false);
            };

            // Start consuming
            channel.BasicConsume(queue: queue, autoAck: false, consumer: consumer);
        }

        public Task StartAsync(CancellationToken cancellationToken)
        {

            _logger.LogInformation(message: $"RabitMQConsumer starting, rabbitmqUri: {_uri}");

            Consume("queue.dataFace", (o, msg) =>
            {
                string[] parts = msg.Split(",");
                string userId = parts[0];
                string message = parts[1];

                _hubContext.Clients.Client(userId).SendAsync("messageReceived", "queue.dataFace", msg);
            });

            Consume("queue.dataFace1", (o, msg) =>
            {
                _hubContext.Clients.All.SendAsync("messageReceived", "queue.dataFace1", msg);
            });

            _logger.LogInformation("RabbitMQ Consumer started");

            return Task.CompletedTask;
        }

        public Task StopAsync(CancellationToken cancellationToken)
        {
            channel.Dispose();
            connection.Dispose();

            _logger.LogInformation("RabitMQConsumer stopping");
            return Task.CompletedTask;
        }

        public void Dispose()
        {
            channel.Dispose();
            connection.Dispose();
        }
    }
}
