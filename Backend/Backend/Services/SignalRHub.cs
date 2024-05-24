using Microsoft.AspNetCore.SignalR;

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
            var userId = Context.ConnectionId;

            _logger.LogInformation($"New message from {userId}: {message}");

            if (!string.IsNullOrEmpty(userId))
            {
                _rabbitMQProducer.SendMessageDirect("queue.dataFace", userId + "," + message);
                //await Clients.Client(userId).SendAsync("messageReceived", userId, message);
            }
        }
    }
}
