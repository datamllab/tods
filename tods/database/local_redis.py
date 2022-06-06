import redis

r = redis.Redis(
    host='localhost',
    port=6381, 
    password='foobar')

r.set('foo', 'bar')

def connect_to_redis(host, port, password):
    r = redis.Redis(host = host, port = port, password = password)
    return r

def 