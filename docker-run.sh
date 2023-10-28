# pull latest changes from git repo
git submodule update --init --recursive

# build docker image
docker build --progress=plain -t vizard:latest -f Dockerfile-dev .

# run docker image
docker run -it -p 5000:5000 -p 8000:8000 vizard:latest

# find the container running `vizard:latest` image
rc=$(docker ps | grep -oP '.*\s\K\w+$' | tail -n 1)

# execute API inference command inside `vizard:latest` running container
docker exec -it $rc mamba run --no-capture-output -n viz-inf-cpu python -m api.main \
--experiment_name "test" \
--run_id "\$1" \
--bind 0.0.0.0 \
--gunicorn_port 8000 \
--mlflow_port 5000 \
--verbose debug \
--workers 1
