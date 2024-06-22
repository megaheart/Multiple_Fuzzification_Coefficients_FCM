using Backend.Enums;
using Microsoft.AspNetCore.SignalR;
using RabbitMQ.Client;
using RabbitMQ.Client.Events;
using System.Diagnostics;
using System.Text;
using System.Text.Json;

namespace Backend.Services
{
    public class RabbitMQConsumer : IDisposable, IHostedService
    {
        private readonly string _uri;
        private readonly ConnectionFactory factory;
        private readonly ILogger<RabbitMQConsumer> _logger;
        private readonly IHubContext<SignalRHub> _hubContext;
        private IConnection connection;
        private IModel channel;

        public RabbitMQConsumer(ILogger<RabbitMQConsumer> logger, IConfiguration configuration, IHubContext<SignalRHub> hubContext)
        {
            _uri = configuration.GetConnectionString("RabbitMQ") ?? throw new Exception("rabbitmqUri phải khác null");
            factory = new ConnectionFactory() { Uri = new Uri(_uri) };
            _logger = logger;
            _hubContext = hubContext;
        }

        public async Task Listen()
        {
            bool notConnected = true;
            while (notConnected)
            {
                try
                {
                    connection = factory.CreateConnection();
                    channel = connection.CreateModel();
                    notConnected = false;
                }
                catch (Exception ex)
                {
                    _logger.LogError("RabbitMQ connection error, retry in 5s");
                    Debug.WriteLine(ex.Message);
                    await Task.Delay(5000);
                }
            }


            Consume(QueueNames.Server, (o, msg) =>
            {
                var json = JsonSerializer.Deserialize<Dictionary<string, object>>(msg);

                _hubContext.Clients.Client(json["connId"].ToString() ?? "").SendAsync("messageReceived", "queue.dataFace", msg);
            });

            //Consume("queue.dataFace", (o, msg) =>
            //{
            //    string[] parts = msg.Split(",");
            //    string userId = parts[0];
            //    string message = parts[1];

            //    _hubContext.Clients.Client(userId).SendAsync("messageReceived", "queue.dataFace", msg);
            //});

            //Consume("queue.dataFace1", (o, msg) =>
            //{
            //    _hubContext.Clients.All.SendAsync("messageReceived", "queue.dataFace1", msg);
            //});

            _logger.LogInformation("RabbitMQ Consumer started");
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

        public void ConsumeBinary(string queue, EventHandler<byte[]> eventHandler)
        {

            // Declare queue
            channel.QueueDeclare(queue: queue, durable: false, exclusive: false, autoDelete: false, arguments: null);

            // Create consumer
            var consumer = new EventingBasicConsumer(channel);
            consumer.Received += (model, ea) =>
            {
                var body = ea.Body.ToArray();
                eventHandler(this, body);
                _logger.Log(LogLevel.Information, $"RabbitMQ bin msg: length={body?.Length ?? 0}");
                channel.BasicAck(deliveryTag: ea.DeliveryTag, multiple: false);
            };

            // Start consuming
            channel.BasicConsume(queue: queue, autoAck: false, consumer: consumer);
        }

        public async Task StartAsync(CancellationToken cancellationToken)
        {
            _logger.LogInformation(message: $"RabitMQConsumer starting, rabbitmqUri: {_uri}");

            Listen();
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
