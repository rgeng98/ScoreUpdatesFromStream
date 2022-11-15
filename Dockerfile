FROM ubuntu:20.04
RUN apt update
COPY . .
RUN apt-get update && \
    apt-get install -y python3.8 python3-pip
RUN pip3 install --upgrade pip
RUN pup3 install --upgrade Pillow
RUN pip3 install torch torchvision IPython numpy 
RUN apt-get install s3fs -y
RUN chmod 600 .passwd-s3fs
RUN sed  -i '/user_allo_other/s/^#//g' /etc/fuse.conf
# RUN /usr/bin/s3fs Training s3-mnt/ -o passwd_file=.passwd-s3fs
CMD ["./RunFile.sh"]