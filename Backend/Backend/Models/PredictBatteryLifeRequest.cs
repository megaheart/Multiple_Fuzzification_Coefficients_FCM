namespace Backend.Models
{
    public class PredictBatteryLifeRequest
    {
        public int[] SupervisedBatteryOrders { get; set; }
        public double[] PredictingState { get; set; }
        public int PredictingBatteryOrder { get; set; }
        public int PredictingCycleOrder { get; set; }
    }
}
