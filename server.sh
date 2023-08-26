#!/bin/sh
nohup python -u app.py > server.out 2>&1 &
tail -f server.out
