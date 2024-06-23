using AutoMapper;
using Backend.Models;

namespace Backend.Mappers
{
    public class MappingProfile : Profile
    {
        public MappingProfile()
        {
            CreateMap<BatteryCycleState, BatteryCycleStateDTO>();
            CreateMap<PredictBatteryLifeRequest, PredictBatteryLifeAiServerRequest>()
                .ForAllMembers(opt => 
                    opt.Condition((source, dest, sourceMember, destMember) => (sourceMember != null))); ;
            CreateMap<PredictBatteryLifeAiServerResponse, PredictBatteryLifeResponse>();
        }
    }
}
