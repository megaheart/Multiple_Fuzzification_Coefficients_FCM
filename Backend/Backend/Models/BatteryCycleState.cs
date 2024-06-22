namespace Backend.Models
{
    public class BatteryCycleState
    {
        public int battery_order { get; init; }
        public int cycle_order { get; init; }
        public double c1a_I_dt { get; init; }
        public double c1a_avg_T { get; init; }
        public double c1a_avg_I { get; init; }
        public double c1_max_I { get; init; }
        public double c2_max_I { get; init; }
        public double c1_max_T { get; init; }
        public double c1_min_T { get; init; }
        public double c2_max_T { get; init; }
        public double c2_min_T { get; init; }
        public double Qi { get; init; }
        public BatteryCycleState(int battery_order, int cycle_order, double c1a_I_dt, double c1a_avg_T, double c1a_avg_I, 
            double c1_max_I, double c2_max_I, double c1_max_T, double c1_min_T, double c2_max_T, double c2_min_T, double Qi)
        {
            this.battery_order = battery_order;
            this.cycle_order = cycle_order;
            this.c1a_I_dt = c1a_I_dt;
            this.c1a_avg_T = c1a_avg_T;
            this.c1a_avg_I = c1a_avg_I;
            this.c1_max_I = c1_max_I;
            this.c2_max_I = c2_max_I;
            this.c1_max_T = c1_max_T;
            this.c1_min_T = c1_min_T;
            this.c2_max_T = c2_max_T;
            this.c2_min_T = c2_min_T;
            this.Qi = Qi;
        }
    }
}
