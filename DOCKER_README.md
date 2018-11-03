## Installation Requirements

**Goal:** Solve deep learning problem using the docker container (Deepo) + python 2.7 (or python 3) as a language and a jupyter notebook as a IDE. The overall code has to be able to run on GPU if available.

This challenge is going to be developed using a jupyter notebook running from a docker container (deepo) allowing us to use different deep learning libraries.  To be able to use the GPU the package nvidia-docker has been used. <br>

**Install Docker on Ubuntu (14.04,16.04,18.04):**

Official documentation: https://docs.docker.com/install/linux/docker-ce/ubuntu/ and https://askubuntu.com/questions/938700/how-do-i-install-docker-on-ubuntu-16-04-lts    

- Set up the docker repository
> sudo apt-get update <br> sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
<br> curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
<br> sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
- Install docker-ce:
> sudo apt-get update <br> sudo apt-get install docker-ce
- Run hello-world container example:
> sudo docker run hello-world

**Install nvidia-docker 2.0 on Ubuntu (14.04,16.04,18.04):**

Official documentation: https://github.com/NVIDIA/nvidia-docker

- If you have nvidia-docker 1.0 installed: we need to remove it and all existing GPU containers
> sudo docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
<br> sudo apt-get purge -y nvidia-docker
- Add the package repositories
<br>
> curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey sudo apt-key add -
<br> distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)
<br> curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list sudo tee /etc/apt/sources.list.d/nvidia-docker.list
<br> sudo apt-get update

- Install nvidia-docker2 and reload the Docker daemon configuration
> sudo apt-get install -y nvidia-docker2 <br> sudo pkill -SIGHUP dockerd

- Test nvidia-smi with the latest official CUDA image
> sudo docker run --runtime=nvidia --rm nvidia/cuda:9.0-base nvidia-smi


** Install Deepo (Deep learning container) from DockerHub**
Official documentation: https://github.com/ufoym/deepo
- Pull docker Deepo Image ufoym/deepo:all-py27-jupyter  
> sudo docker pull ufoym/deepo:all-py27-jupyter
- Create and run the container, given the pulled image, using nvidia-docker to have GPU support
> sudo nvidia-docker run --rm ufoym/deepo:all-py27-jupyter nvidia-smi
- Run the container with the following features: <br>
1) -p 8888:8888 jupyter notebook available at localhost:8888 <br>
2) -v /home/$USER/notebooks:/root means that the folder /home/$USER/notebooks is our share folder. In this folder we can place the data of our kaggle challage such that they are visible inside the container.
> sudo nvidia-docker run -it -p 8888:8888 -v /home/$USER/notebooks:/root ufoym/deepo:all-py27-jupyter jupyter notebook --ip=0.0.0.0 --allow-root --NotebookApp.token= --notebook-dir='/root'</dd>

**Make your jupyter notebook better**
- Stop the running container. To do this u need docker  CONTAINER ID
> sudo docker ps  (show us the running containers) to see the both running and stopped ones type sudo docker ps -a

- Using the CONTAINER ID stop the container
> sudo docker stop CONTAINER ID (example sudo docker stop b73a00aef4d0 )

- Start the container again in iterative mode
> sudo docker exec -it b73a00aef4d0 bash

- Install jupyter notebook extension (Official documentation: https://github.com/ipython-contrib/jupyter_contrib_nbextensions)
> pip install --user jupyter_contrib_nbextensions  (u need to put --user otherwise u have wrong path matching)<br>
jupyter nbextensions_configurator enable --user (enable jupyter configuration server)

- Exit and Stop container
> exit

- When u switch on the container again check at localhost:8888/nbextensions , now u can configure your notebook and make it prettier.



### Fix permissions
Fix permissions: https://stackoverflow.com/questions/48957195/how-to-fix-docker-got-permission-denied-issue
sudo groupadd docker <br>
sudo usermod -aG docker $USER <br>
docker run hello-world <br>
sudo systemctl restart docker (maybe u also need to reboot) <br>

reference: https://askubuntu.com/questions/747778/docker-warning-config-json-permission-denied <br>
sudo chown "$USER":"$USER" /home/"$USER"/.docker -R <br>
sudo chmod g+rwx "/home/$USER/.docker" -R <br>

Entra nel container <br>
docker ps <br>
docker exec -it b73a00aef4d0 bash <br>
cd root <br>
ls -l (look at the numbers) <br>
chown -R 1000:1000 file <br>
