FROM aspnet-python-rabbitmq:latest AS base
WORKDIR /app
COPY ["./AI/install_pkg.sh", "./AI/"]
RUN chmod +x ./AI/install_pkg.sh && \
    ./AI/install_pkg.sh

# Get  
RUN apt-get update && apt-get install -y python3-pip

FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
ARG BUILD_CONFIGURATION=Release
WORKDIR /src
COPY ["./Backend/Backend/Backend.csproj", "Backend/"]
RUN dotnet restore "./Backend/Backend.csproj"
COPY ["./Backend/Backend", "Backend/"]
WORKDIR "/src/Backend"
RUN dotnet build "./Backend.csproj" -c $BUILD_CONFIGURATION -o /app/build

FROM build AS publish
ARG BUILD_CONFIGURATION=Release
RUN dotnet publish "./Backend.csproj" -c $BUILD_CONFIGURATION -o /app/publish /p:UseAppHost=false

FROM base AS server
WORKDIR /app

COPY --from=publish /app/publish ./Backend
# COPY ["./AI/requirements.txt", "./AI/"]
COPY ["./Backend/Backend/wwwroot", "./Backend/wwwroot"]

COPY ["./AI/Components", "./AI/Components"]
COPY ["./AI/ai_server.py", "./AI/"]

EXPOSE 80
EXPOSE 443
EXPOSE $PORT

# Start the server
COPY ["./init_container.sh", "./"]
RUN chmod +x ./init_container.sh
CMD ["./init_container.sh"]