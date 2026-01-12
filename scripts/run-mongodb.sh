#!/bin/bash
docker run -d --name mongodb -p 27017:27017 -v $(pwd)/data/mongodb:/data/db mongo:latest
