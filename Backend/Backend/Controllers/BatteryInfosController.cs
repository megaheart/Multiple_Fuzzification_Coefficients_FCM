using AutoMapper;
using Backend.Helpers;
using Backend.Models;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Caching.Memory;
using System.Collections.Immutable;

namespace Backend.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class BatteryInfosController : ControllerBase
    {
        private readonly IMemoryCache _cache;
        private readonly IMapper _mapper;

        public BatteryInfosController(IMemoryCache cache, IMapper mapper)
        {
            _cache = cache;
            _mapper = mapper;
        }

        [HttpGet]
        public ActionResult Get()
        {
            if (!_cache.TryGetValue("batteryInfoList", out ImmutableList<BatteryInfo>? batteryInfoList))
            {
                return NotFound();
            }

            return Ok(batteryInfoList ?? new List<BatteryInfo>().ToImmutableList());
        }

        [HttpGet("states")]
        public ActionResult GetStates([FromQuery] string batteryOrders)
        {
            List<int> _batteryOrders;
            try 
            {
                _batteryOrders = batteryOrders.Split(',').Select(int.Parse).ToList();
            }
            catch (Exception)
            {
                return BadRequest("Định dạng batteryOrders không hợp lệ");
            }

            if (_batteryOrders == null || _batteryOrders.Count == 0)
            {
                return BadRequest("batteryOrders không được để trống");
            }
            var batteryStates = new List<BatteryCycleState>();
            foreach (var batteryOrder in _batteryOrders)
            {
                if (!_cache.TryGetValue($"batteryCycleStateList_b{batteryOrder}", out ImmutableList<BatteryCycleState>? batteryCycleStates) 
                    || batteryCycleStates == null)
                {
                    return NotFound();
                }
                batteryStates.AddRange(batteryCycleStates);
            }

            batteryStates = batteryStates.ChooseRandom(100);

            var result = _mapper.Map<List<BatteryCycleStateDTO>>(batteryStates);

            return Ok(result ?? new List<BatteryCycleStateDTO>());
        }
    }
}
