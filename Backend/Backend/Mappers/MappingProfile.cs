using AutoMapper;
using Backend.Models;

namespace Backend.Mappers
{
    public class MappingProfile : Profile
    {
        public MappingProfile()
        {
            CreateMap<BatteryCycleState, BatteryCycleStateDTO>();
        }
    }
}
