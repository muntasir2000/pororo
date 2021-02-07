git pull origin master
sudo docker build -t po .
sudo docker run -it -v /2hdd/temp/models:/models -v $(pwd):/app po bash