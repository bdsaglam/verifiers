# dstack

https://dstack.ai/docs


## Server

We can either use [dstack Sky](https://sky.dstack.ai/) or local server. 

### Local server
[Server deployment docs](https://dstack.ai/docs/guides/server-deployment/)

Run the following command to start the local server:

```
docker run -p 3000:3000 \
    -v $HOME/.dstack/server/:/root/.dstack/server \
    dstackai/dstack
```

This will start the server at `http://localhost:3000` and print the token.

Configure backends at `~/.dstack/server/config.yml`. For instance, for Lambda backend:

```
projects:
- name: main
  backends:
    - type: lambda
      creds:
        type: api_key
        api_key: LAMBDA_API_KEY

```

Configure the CLI:

```sh
dstack config --url localhost:3000 --project bdsaglam --token $DSTACK_TOKEN
```

### Sky server

```sh
dstack config --url https://sky.dstack.ai --project bdsaglam --token $DSTACK_TOKEN
```

## Tasks

### Batch predict

```sh
dstack apply -f dstacks/predict.dstack.yml 
```