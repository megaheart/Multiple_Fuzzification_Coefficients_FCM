using Microsoft.AspNetCore.Connections;
using RabbitMQ.Client;
using System;
using System.Text;
using static System.Runtime.InteropServices.JavaScript.JSType;
using System.Text.Json;

namespace Backend.Services
{
    public class RabbitMQProducer
    {
        private readonly string _uri;
        private readonly ConnectionFactory factory;
        private readonly ILogger<RabbitMQConsumer> _logger;
        public RabbitMQProducer(ILogger<RabbitMQConsumer> logger, IConfiguration configuration)
        {
            _uri = configuration.GetConnectionString("RabbitMQ") ?? throw new Exception("rabbitmqUri phải khác null");
            factory = new ConnectionFactory() { Uri = new Uri(_uri) };
        }

        public void SendMessageDirect(string queue, string message)
        {
            using (var connection = factory.CreateConnection())
            using (var channel = connection.CreateModel())
            {
                // Declare queue
                channel.QueueDeclare(queue: queue, durable: false, exclusive: false, autoDelete: false, arguments: null);

                // Convert message to byte array
                var body = Encoding.UTF8.GetBytes(message);

                // Publish data to RabbitMQ broker
                //channel.BasicPublish(exchange: "", routingKey: "hello", basicProperties: null, body: body);
                channel.BasicPublish(exchange: "", routingKey: queue, basicProperties: null, body: body);
            }

        }
    }
}
