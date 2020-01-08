# btgym docker for CPU only

## To Build
- sudo docker image build -t btgym .

## To Run
- docker container run -p 8080:8080 -p 6006:6006 -p 8888:8888 btgym

## To View
- [Desktop] point your browser to http://127.0.0.1:8080/vnc.html (or external IP), the password has been set to btgym
- [Tensorboard] browse http://127.0.0.1:6006 (or external IP)

### To link local directory
- add "-v /local_dir:/container_dir"

### To use your own copy of btgym
- comment out the following line in entrypoint.sh
exec git clone https://github.com/Kismuz/btgym.git /workspace &

## Notes
- The container starts with chromium opening the jupyter nootbook authorization, select the file nbserver-#-open.html to access the notbook server.
- Jupyter is also configured to accept connections from Google Collab

##### Original credits to:
- https://github.com/theasp/docker

##### Which was based on:
- https://github.com/psharkey/docker/tree/master/novnc
