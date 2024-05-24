
using Backend.Services;
using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using System.Text.Json;
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


            builder.Services.AddSignalR();

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

            app.Run();


        }
    }
}
