import pika
import sys
import ssl
import binascii
import json
import time

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

channel = connection.channel()
channel.queue_declare(queue='queue.ai')
channel.queue_declare(queue='queue.server')
# channel.queue_declare(queue='queue.dataFace')


def callback(ch, method, properties, body):
    print(" [x] Received %r" % body)
    dict = json.loads(body)
    print(dict['name'], dict['age'], dict)
    channel.basic_publish(exchange='', routing_key='queue.server', body=body)
    ch.basic_ack(delivery_tag = method.delivery_tag)

channel.basic_qos(prefetch_count=1)
channel.basic_consume(on_message_callback=callback, queue='queue.ai', auto_ack=False)
print("Start consuming...")
channel.start_consuming()