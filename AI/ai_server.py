import pika
import sys
import ssl
import binascii
import json
import time
import redis
import procedures as proc

connection = None

while connection is None:
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(
            host='localhost',
            port=5672,
            virtual_host='/',
            credentials=pika.PlainCredentials('admin1','123456'),
            heartbeat=0 
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
    supervisedBatteryOrders = req['SupervisedBatteryOrders']
    # predictingState = req['PredictingState']
    predictingBatteryOrder = req['PredictingBatteryOrder']
    # predictingCycleOrder = req['PredictingCycleOrder']
    connectionId = req['ConnectionId']

    # check if the key exists
    if not redisDb.exists("conn:" + req['ConnectionId']):
        print("Key does not exist")
        ch.basic_ack(delivery_tag = method.delivery_tag)
        return
    
    print("Key = conn:", req['ConnectionId'], ", Value = ", redisDb.get("conn:" + req['ConnectionId']))

    # Response to server
    # res = {
    #     "isSuccessful": True,
    #     "connectionId": req['ConnectionId'],
    #     "type": "PredictingQi",
    #     "message": "Success",
    #     "value": [] # list of double values
    # }
    # res = json.dumps(res)
    res = proc.start_predict_capacity_response(req['ConnectionId'])
    channel.basic_publish(exchange='', routing_key='queue.server', body=res)
    print(" [x] Sent %r" % res)
    dataframe, true_capacity, capacity = proc.predict_capacity(supervisedBatteryOrders, predictingBatteryOrder)
    res = proc.start_predict_remain_life_response(req['ConnectionId'])
    channel.basic_publish(exchange='', routing_key='queue.server', body=res)
    print(" [x] Sent %r" % res)
    true_remain_life, remain_life = proc.predict_remain_life(dataframe, supervisedBatteryOrders, predictingBatteryOrder, capacity)
    res = proc.finish_predict_response(req['ConnectionId'], dataframe, predictingBatteryOrder, capacity, remain_life, true_capacity, true_remain_life)
    channel.basic_publish(exchange='', routing_key='queue.server', body=res)
    print(" [x] Sent %r" % res)

    # ACK
    ch.basic_ack(delivery_tag = method.delivery_tag)
    redisDb.delete("conn:" + req['ConnectionId'])
    print("Process Completed")

channel.basic_qos(prefetch_count=1)
channel.basic_consume(on_message_callback=callback, queue='queue.ai', auto_ack=False)
print("Start consuming...")
channel.start_consuming()