import os
from redis import Redis
from rq import Worker, Queue

def main():
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    conn = Redis.from_url(redis_url)

    q = Queue("default", connection=conn)
    worker = Worker([q], connection=conn)
    worker.work(with_scheduler=False)

if __name__ == "__main__":
    main()
