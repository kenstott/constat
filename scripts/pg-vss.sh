#!/usr/bin/env bash
docker run -d --name constat-pg -p 5432:5432 -e POSTGRES_PASSWORD=constat -e POSTGRES_DB=constat pgvector/pgvector:pg17
