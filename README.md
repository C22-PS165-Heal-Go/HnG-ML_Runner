# Heal&Go ML Runner

ML parts runner as an internal service 

## Built Using

- Python
- FastAPI
- Uvicorn
- Tensorflow

## TODO

- [ ] Implement DL Model
- [ ] Implement RDL Model
- [ ] Dockerized 

## Models
models are submodule which linked to ML's Repo(https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning)

## Development

### Requirements

- Python 3

### Getting Started

- Pull this repo
- Run `python -m uvicorn main:app --reload --host 0.0.0.0 --port 5001` to start developing


## Deployment

### Docker (Coming soon)

- Clone this repo to target machine
- Use provided docker-compose here
- Run `./deploy.sh` to start using default config
- Setup a reverse proxy for default port 5001

