AI-Tutor Testing

Install required dependencies for 'requirements.txt' by running following command

>> pip install -r requirements.txt

Run FastAPI Server by using following command in your terminal

To run testing.py -

>> uvicorn testing:app --reload

To run tutor.py -

>> uvicorn tutor:app --reload

Server will respond according to type of data feeded to it.
It cannot answer the questions which are out of context.
