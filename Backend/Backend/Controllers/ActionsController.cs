using Backend.Services;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using StackExchange.Redis;

namespace Backend.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ActionsController : ControllerBase
    {
        private readonly IDatabase _redis;
        private readonly ILogger<SignalRHub> _logger;

        public ActionsController(IConnectionMultiplexer multiplexer, ILogger<SignalRHub> logger)
        {
            _redis = multiplexer.GetDatabase(0);
            _logger = logger;
        }

        [HttpDelete("flush-redis-cache")]
        public IActionResult FlushRedisCache()
        {
            var result = _redis.Execute("FLUSHALL");
            return Ok(result);
        }

        [HttpPost("command-redis-cache")]
        public IActionResult CommandRedisCache([FromBody] string command)
        {
            try
            {
                var result = _redis.Execute(command);
                return Ok(result);
            }
            catch (RedisServerException ex)
            {
                _logger.LogError(ex, "Error executing command");
                return BadRequest("");
            }
        }
    }
}
