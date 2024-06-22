namespace Backend.Models
{
    public class BatteryInfo
    {
        public int battery_order { get; init; }
        public string policy_readable { get; init; }
        public int cycle_count { get; init; }

        public BatteryInfo(int battery_order, string policy_readable, int cycle_count)
        {
            this.battery_order = battery_order;
            this.policy_readable = policy_readable;
            this.cycle_count = cycle_count;
        }

    }
}
