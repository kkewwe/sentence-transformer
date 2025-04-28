## Sentence Transformer(s) & Multi-Task Learning


### About Me: 

This project implements a sentence transformer model that encodes input sentences into fixed-length embeddings. 
The applications of such sentence transformers could be for tasks such as semantic chunking, text similarity, etc...

**Input:** a sentence/string. 

**Output:** a fixed-size numerical embedding vector for each sentence.

**Model:** all-MiniLM-L6-v2 (https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

The application uses Docker as well as FastAPI with a Swagger page to allow for easy testing.   


### Quick Start:

Recommended to be run in Python 3.11, as dependencies are configured to that version. There are two possible ways to 
start the application. Either through Docker (recommended) or through command line python commands. 

#### 1A. To run using Docker: 
```commandLine
docker-compose up --build
```

#### 1B. To run using Python: 
```commandline
pip install -r requirements.txt
python app/main.py
uvicorn api:app --reload
```

#### 2. FastAPI Swagger interface: 
In your browser navigate to: ```http://localhost:8000/docs#/```, it should look like this: 
![img.png](img.png)
In each endpoint you can test each task individually by expanding the endpoint, selecting "Try it Out", then inputting 
a sentence value to replace the existing "sentence" default value. The request will need to be in valid JSON so ensure
that quotes and curly braces are in the correct place!
![img_1.png](img_1.png)

There is also an optional dimension field. The default dimensions is 384, but if the user specifies a dimension, the
program will project embeddings before feeding them into the models. This is a simple compression, so it's not 
semantically perfect. If I were to want to use a better dimension reduction, I would use PCA in order to perform it. 

#### 3. Running the test file: 
```commandline
python test.py
```

### Deep Dive into Task 1:

I implemented a sentence transformer model, encoding input sentences into fixed-length dense embeddings. I used a 
pre-trained model named all-MiniLM-L6-v2 through the SentenceTransformer library. This library automatically tokenizes
the text, passes it into the Transformer model, then applies mean pooling to produce the single vector. 

I also implemented a dynamic projection layer (nn.Linear) that would allow users to choose embedding dimensions outside
of the default 384. Again, this is a simple projection and a more robust solution like PCA would be something I would 
recommend for scalability. 

#### Test One: 

#### Test Two: 

#### Test Three: 


### Deep Dive into Task 2:




