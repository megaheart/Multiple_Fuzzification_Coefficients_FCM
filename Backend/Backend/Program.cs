
using Backend.Mappers;
using Backend.Services;
using CsvHelper;
using CsvHelper.Configuration;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Caching.Memory;
using StackExchange.Redis;
using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace Backend
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            var configuration = builder.Configuration;

            // Add services to the container.
            builder.Services.AddScoped<RabbitMQProducer>();


            builder.Services.AddSignalR()
                .AddJsonProtocol(options =>
                {
                    options.PayloadSerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.CamelCase;
                    options.PayloadSerializerOptions.Converters.Add(new JsonStringEnumConverter());
                });

            builder.Services.AddControllers().ConfigureApiBehaviorOptions(options =>
            {
                options.InvalidModelStateResponseFactory = context =>
                {
                    var errors = new Dictionary<string, IEnumerable<string>>();
                    foreach (var pair in context.ModelState)
                    {
                        var key = pair.Key;
                        var value = pair.Value.Errors.Select(e => e.ErrorMessage);
                        if (value != null)
                        {
                            errors.Add(key, value);
                        }
                    }
                    return new BadRequestObjectResult(new Dictionary<string, object>()
                    {
                        ["UserMsg"] = "Invalid model states.",
                        ["DevMsg"] = "Invalid model states.",
                        ["MoreInfo"] = "Invalid model states.",
                        ["TraceId"] = context.HttpContext.TraceIdentifier,
                        ["Data"] = errors
                    });
                };
            })
            .AddJsonOptions(options =>
            {
                options.JsonSerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.CamelCase;
            });

            builder.Services.AddCors(options =>
            {
                options.AddPolicy("AllowAll", builder =>
                {
                    builder.AllowAnyOrigin()
                           .AllowAnyMethod()
                           .AllowAnyHeader();

                    //.SetIsOriginAllowed(origin => true)
                    //.AllowCredentials();
                });
            });


            // Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
            builder.Services.AddEndpointsApiExplorer();
            builder.Services.AddSwaggerGen();
            builder.Services.AddMemoryCache();
            //builder.Services.AddStackExchangeRedisCache(options => {
            //    options.Configuration = builder.Configuration.GetConnectionString("Redis");
            //});

            builder.Services.AddSingleton<IConnectionMultiplexer>(ConnectionMultiplexer.Connect(
                builder.Configuration.GetConnectionString("Redis") ?? throw new Exception("Redis Connection String doesn't exist.")));

            builder.Services.AddAutoMapper((serviceProvider, automapper) =>
            {
                automapper.AddProfile<MappingProfile>();
            }, AppDomain.CurrentDomain.GetAssemblies());
            builder.Services.AddHostedService<RabbitMQConsumer>();

            var app = builder.Build();

            var logger = app.Services.GetRequiredService<ILogger<Program>>();

            // Configure the HTTP request pipeline.
            if (app.Environment.IsDevelopment())
            {
                app.UseSwagger();
                app.UseSwaggerUI();
            }

            app.UseHttpsRedirection();

            app.UseDefaultFiles();
            app.UseStaticFiles();
            app.UseRouting();

            app.UseCors("AllowAll");

            app.UseAuthorization();

            app.MapControllers();

            app.MapHub<SignalRHub>("/hub");

            app.MapFallbackToFile("index.html");

            app.LoadFilesToCache();

            app.Run();


        }
    }
}
