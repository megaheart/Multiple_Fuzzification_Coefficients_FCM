using Backend.Models;

namespace Backend.Interfaces
{
    public interface ISignalrClient
    {
        Task MessageReceived(string message);
        Task PredictProgress(PredictBatteryLifeResponse response);
    }
}
