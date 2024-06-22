using CsvHelper.Configuration;
using CsvHelper;
using Microsoft.Extensions.Caching.Memory;
using System.Globalization;
using Backend.Models;
using System.Collections.Immutable;

namespace Backend.Services
{
    public static class LoadFilesToCacheExtension
    {
        public static void LoadFilesToCache(this WebApplication app)
        {
            // Load Csv file to cache
            var cache = app.Services.GetRequiredService<IMemoryCache>();
            var config = new CsvConfiguration(CultureInfo.InvariantCulture)
            {
                HasHeaderRecord = true,
            };
            ImmutableList<BatteryInfo> batteryInfoList;
            using (var reader = new StreamReader("./Resources/battery_infos.csv"))
            using (var csv = new CsvReader(reader, config))
            {
                batteryInfoList = csv.GetRecords<BatteryInfo>().ToImmutableList();
            }
            cache.Set("batteryInfoList", batteryInfoList);

            ImmutableList<BatteryCycleState> batteryCycleStateList;
            using (var reader = new StreamReader("./Resources/battery_cycles.csv"))
            using (var csv = new CsvReader(reader, config))
            {
                batteryCycleStateList = csv.GetRecords<BatteryCycleState>().ToImmutableList();
            }
            
            for (int i = 0; i < batteryInfoList.Count; i++)
            {
                var batteryCycleStates = batteryCycleStateList
                    .Where(b => b.battery_order == batteryInfoList[i].battery_order).ToImmutableList();
                cache.Set($"batteryCycleStateList_b{batteryInfoList[i].battery_order}", batteryCycleStates);
            }
        }
    }
}
