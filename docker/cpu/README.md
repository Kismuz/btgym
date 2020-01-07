# btgym docker for CPU only

## To Build
- sudo docker image build -t btgym .

## To Run
- docker container run -p 8080:8080 -p 6006:6006 -p 8888:8888 btgym

### To link local directory
- add "-v /local_dir:/container_dir"

### To use your own copy of btgym
- comment out the following line in entrypoint.sh
exec git clone https://github.com/Kismuz/btgym.git /workspace &

## Notes
The container starts with chromium opening the jupyter nootbook authorization, select the file nbserver-#-open.html to access the notbook server.

##### Original credits to:
- https://github.com/theasp/docker

##### Which was based on:
- https://github.com/psharkey/docker/tree/master/novnc
