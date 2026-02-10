from fastapi import FastAPI,Query
from client.rq_client import queue
from queues.worker import process_query

app = FastAPI()

@app.get("/")
def root( ):
    return {"status":"server is running"}




@app.post("/chat")
def chat(
        query : str = Query(... , description="the chat user query")
):
    job = queue.enqueue(process_query,query)  
    return {"status":"queued","jobid":job.id}

@app.get("/get_results")
def get_results(
        job_id : str = Query(... , description="Job id")

        ):
    job = queue.fetch_job(job_id=job_id)

    results = job.return_value

    return {"results":results}