using Backend.Enums;
using Backend.Models;
using Microsoft.AspNetCore.SignalR;
using Microsoft.Extensions.Primitives;
using Microsoft.VisualBasic;
using System.Text;
using System.Text.Json;
using Backend.Enums;
using Backend.Interfaces;
using Microsoft.Extensions.Caching.Distributed;
using AutoMapper;
using StackExchange.Redis;

namespace Backend.Services
{
    public class SignalRHub : Hub<ISignalrClient>
    {
        private readonly IHubContext<SignalRHub> _hubContext;
        private readonly ILogger<SignalRHub> _logger;
        private readonly RabbitMQProducer _rabbitMQProducer;
        private readonly IDatabase _redis;
        private readonly IMapper _mapper;

        public SignalRHub(IHubContext<SignalRHub> hubContext, ILogger<SignalRHub> logger, RabbitMQProducer rabbitMQProducer, IConnectionMultiplexer multiplexer, IMapper mapper)
        {
            _hubContext = hubContext;
            _logger = logger;
            _rabbitMQProducer = rabbitMQProducer;
            _redis = multiplexer.GetDatabase(0);
            _mapper = mapper;
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
                //|| request.PredictingState == null || request.PredictingState.Length == 0 || request.PredictingCycleOrder == 0
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
                var aiServerRequest = _mapper.Map<PredictBatteryLifeAiServerRequest>(request);
                aiServerRequest.ConnectionId = connId;
                try
                {
                    await _rabbitMQProducer.SendJsonDirect(QueueNames.AI, aiServerRequest);

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

        public override async Task OnConnectedAsync()
        {
            var connId = Context.ConnectionId;
            _logger.LogInformation($"New connection from {connId}");
            await _redis.StringSetAsync("conn:" + connId, "connected");
            await base.OnConnectedAsync();
        }

        public override async Task OnDisconnectedAsync(Exception exception)
        {
            var connId = Context.ConnectionId;
            _logger.LogInformation($"Disconnected from {connId}");
            await _redis.KeyDeleteAsync("conn:" + connId);
            await base.OnDisconnectedAsync(exception);
        }
    }
}
