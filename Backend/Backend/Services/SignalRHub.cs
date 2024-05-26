using Backend.Resources;
using Microsoft.AspNetCore.SignalR;
using Microsoft.VisualBasic;
using System.Text.Json;

namespace Backend.Services
{
    public class SignalRHub : Hub
    {
        private readonly IHubContext<SignalRHub> _hubContext;
        private readonly ILogger<SignalRHub> _logger;
        private readonly RabbitMQProducer _rabbitMQProducer;
        public SignalRHub(IHubContext<SignalRHub> hubContext, ILogger<SignalRHub> logger, RabbitMQProducer rabbitMQProducer)
        {
            _hubContext = hubContext;
            _logger = logger;
            _rabbitMQProducer = rabbitMQProducer;
        }
        public async Task NewMessage(long username, string message)
        {
            // Get Id of user who sent the message
            var connId = Context.ConnectionId;

            _logger.LogInformation($"New message from {connId}: {message}");

            if (!string.IsNullOrEmpty(connId))
            {
                var json = JsonSerializer.Deserialize<Dictionary<string, object>>(message);
                json["connId"] = connId;
                var msg = JsonSerializer.Serialize(json);
                _rabbitMQProducer.SendMessageDirect(QueueNames.AI, msg);
                //await Clients.Client(userId).SendAsync("messageReceived", userId, message);
            }
        }
    }
}
