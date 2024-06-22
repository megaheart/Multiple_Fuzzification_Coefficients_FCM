using Backend.Enums;
using Backend.Models;
using Microsoft.AspNetCore.SignalR;
using Microsoft.Extensions.Primitives;
using Microsoft.VisualBasic;
using System.Text;
using System.Text.Json;
using Backend.Enums;
using Backend.Interfaces;

namespace Backend.Services
{
    public class SignalRHub : Hub<ISignalrClient>
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
        public async Task PredictBatteryLife(PredictBatteryLifeRequest request)
        {
            // Get Id of user who sent the message
            var connId = Context.ConnectionId;
            _logger.LogInformation($"New message from {connId}");
            if (string.IsNullOrEmpty(connId))
            {
                return;
            }
            var client = _hubContext.Clients.Client(connId);
            if (client == null)
            {
                _logger.LogError($"Client not found {connId}");
            }

            if (request == null || request.SupervisedBatteryOrders == null || request.SupervisedBatteryOrders.Length == 0
                || request.PredictingState == null || request.PredictingState.Length == 0 || request.PredictingCycleOrder == 0
                || request.PredictingBatteryOrder == 0)
            {
                var response = new PredictBatteryLifeResponse
                {
                    IsSuccessful = false,
                    Type = PredictBatteryLifeResponseTypes.BadRequest,
                    Message = "Yêu cầu dự đoán vòng đời pin không hợp lệ",
                    Value = null
                };
                client.SendAsync(SignalrEvents.PredictProgress, response);
            }
            else
            {

                var msg = JsonSerializer.Serialize(request);
                try
                {
                    _rabbitMQProducer.SendMessageDirect(QueueNames.AI, msg);

                    var response = new PredictBatteryLifeResponse
                    {
                        IsSuccessful = true,
                        Type = PredictBatteryLifeResponseTypes.PushToQueue,
                        Message = "Tiến hành đẩy yêu cầu dự đoán vòng đời pin vào hàng đợi",
                        Value = null
                    };
                    client.SendAsync(SignalrEvents.PredictProgress, response);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error while sending message to RabbitMQ");
                    var response = new PredictBatteryLifeResponse
                    {
                        IsSuccessful = false,
                        Type = PredictBatteryLifeResponseTypes.BadRequest,
                        Message = "Hàng đợi RabbitMQ chưa sẵn sàng",
                        Value = null
                    };
                    client.SendAsync(SignalrEvents.PredictProgress, response);
                }


            }
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
