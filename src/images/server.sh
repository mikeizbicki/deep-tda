#!/usr/bin/env sh
nohup python -m http.server 8000 &
nohup ~/bin/ngrok http 8000 &
