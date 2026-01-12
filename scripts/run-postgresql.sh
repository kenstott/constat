#!/bin/bash
docker run -d --name postgresql -p 5432:5432 -e POSTGRES_PASSWORD=postgres -v $(pwd)/data/postgresql:/var/lib/postgresql/data postgres:latest
