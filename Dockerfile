FROM aspnet-python-rabbitmq-redis:latest AS base
WORKDIR /app
# Get python3 and pip
# RUN apt-get update && apt-get install -y python3-pip

# Install python packages
# COPY ["./AI/install_pkg.sh", "./AI/"]
# RUN chmod +x ./AI/install_pkg.sh && \
#     ./AI/install_pkg.sh
COPY ["./AI/requirements.txt", "./AI/"]
RUN /opt/py3env/bin/pip install -r ./AI/requirements.txt

# Build the backend
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
ARG BUILD_CONFIGURATION=Release
WORKDIR /src
COPY ["./Backend/Backend/Backend.csproj", "Backend/"]
RUN dotnet restore "./Backend/Backend.csproj"
COPY ["./Backend/Backend", "Backend/"]
WORKDIR "/src/Backend"
RUN dotnet build "./Backend.csproj" -c $BUILD_CONFIGURATION -o /app/build

# Publish the backend
FROM build AS publish
ARG BUILD_CONFIGURATION=Release
RUN dotnet publish "./Backend.csproj" -c $BUILD_CONFIGURATION -o /app/publish /p:UseAppHost=false

# Build angular nodejs frontend
FROM node-pnpm:20.14.0 AS frontend
WORKDIR /src
COPY ["./Frontend/package.json", "./"]
RUN bash -c "npm install"
COPY ["./Frontend/", "./"]
# COPY ["./Frontend/angular.json", "./Frontend/proxy.conf.json", "./Frontend/tsconfig.app.json", "./Frontend/tsconfig.json", "./Frontend/tsconfig.spec.json", "./"]
RUN bash -c "npm run build -- --configuration production --output-path=./dist/frontend/browser"

# Build the final image
FROM base AS server
WORKDIR /app

COPY --from=publish /app/publish ./Backend
# COPY ["./AI/requirements.txt", "./AI/"]
# COPY ["./Backend/Backend/wwwroot", "./Backend/wwwroot"]
COPY --from=frontend /src/dist/frontend/browser/* ./Backend/wwwroot

COPY ["./AI/Components", "./AI/Components"]
COPY ["./AI/Data", "./AI/Data"]
COPY ["./AI/ai_server.py", "./AI/ssmc_fcm.py", "./AI/procedures.py", "./AI/"]

EXPOSE 80
EXPOSE 443
EXPOSE $PORT

# Start the server
COPY ["./init_container.sh", "./"]
RUN chmod +x ./init_container.sh
CMD ["./init_container.sh"]