import pika
import sys
import ssl
import binascii
import json
import time
import redis

connection = None

while connection is None:
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(
            host='localhost',
            port=5672,
            virtual_host='/',
            credentials=pika.PlainCredentials('admin1','123456'),
            # ssl_options=pika.SSLOptions(ssl.create_default_context()),
        ))
    except Exception as e:
        print("Retry connect to RabbitMQ in 5 seconds...")
        time.sleep(5)

redisDb = None

while redisDb is None:
    try:
        redisDb = redis.Redis(
            host='localhost',
            port=6379,
            db=0
        )
    except Exception as e:
        print("Retry connect to Redis in 5 seconds...")
        time.sleep(5)

channel = connection.channel()
channel.queue_declare(queue='queue.ai')
channel.queue_declare(queue='queue.server')
# channel.queue_declare(queue='queue.dataFace')

def callback(ch, method, properties, body):
    global redisDb
    print(" [x] Received %r" % body)
    req = json.loads(body)
    # print(dict['SupervisedBatteryOrders'], dict['PredictingState'], 
    #       dict['PredictingBatteryOrder'], dict['PredictingCycleOrder'], 
    #       dict['ConnectionId'], dict)
    # check if the key exists
    if not redisDb.exists("conn:" + req['ConnectionId']):
        print("Key does not exist")
        ch.basic_ack(delivery_tag = method.delivery_tag)
        return
    
    print("Key = conn:", req['ConnectionId'], ", Value = ", redisDb.get("conn:" + req['ConnectionId']))

    # Response to server
    res = {
        "isSuccessful": True,
        "connectionId": req['ConnectionId'],
        "type": "PredictingQi",
        "message": "Success",
        "value": [] # list of double values
    }
    res = json.dumps(res)
    channel.basic_publish(exchange='', routing_key='queue.server', body=res)
    ch.basic_ack(delivery_tag = method.delivery_tag)
    redisDb.delete("conn:" + req['ConnectionId'])
    print("Process Completed")

channel.basic_qos(prefetch_count=1)
channel.basic_consume(on_message_callback=callback, queue='queue.ai', auto_ack=False)
print("Start consuming...")
channel.start_consuming()