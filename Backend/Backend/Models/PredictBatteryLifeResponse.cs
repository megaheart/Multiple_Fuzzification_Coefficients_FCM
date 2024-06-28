using Backend.Enums;

namespace Backend.Models
{
    public class PredictBatteryLifeResponse
    {
        public bool IsSuccessful { get; set; }
        public PredictBatteryLifeResponseTypes Type { get; set; }
        public string Message { get; set; }
        public double[]? Value { get; set; }
        public double[][]? Values { get; set; }
    }
}
