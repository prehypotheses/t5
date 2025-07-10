<br>

## Development Environment

<br>

> [!IMPORTANT] 
> [Hugging Face Training Containers](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#huggingface-training-containers)<br>
> [Hugging Face Inference Containers](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#huggingface-inference-containers)

<br>

### Beware

For some learning rate schedulers, the number of steps - `TrainingArguments(..., max_steps=)` -  needs to be known in advance.  At present, it is quite possible that the hyperparameter_search() method of Trainer and the <a href="https://pypi.org/project/bayesian-optimization/" target="_blank">`bayesian_optimization` library</a> are incompatible.

For the latest _`backend`_ $\bot$ _`hyperparameter space`_ set up visit the <a href="https://github.com/huggingface/transformers/blob/main/docs/source/en/hpo_train.md" target="_blank">latest hyperparameter opt page</a>.

For checkpoints visit (a) [checkpoints](https://docs.ray.io/en/latest/train/user-guides/checkpoints.html), (b) [checkpoints](https://docs.ray.io/en/latest/train/getting-started-transformers.html#report-checkpoints-and-metrics).

For (a) [distributed training](https://docs.ray.io/en/latest/train/getting-started-transformers.html), (b) [hyperparameter opt](https://docs.ray.io/en/latest/tune/examples/pbt_transformers.html), [alongside huggingface.co hyperparameter search](https://huggingface.co/docs/transformers/v4.53.1/en/hpo_train), [ray.tune.run](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.run.html), [TrainingArguments](https://huggingface.co/docs/transformers/v4.53.1/en/main_classes/trainer#transformers.TrainingArguments), [tune results](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.Result.html), (c) [Optuna Search](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.search.optuna.OptunaSearch.html).

[Fine Tuning](https://docs.ray.io/en/latest/train/examples/transformers/huggingface_text_classification.html)

<br>

### Remote Development

For this Python project/template, the remote development environment requires

* [Dockerfile](../.devcontainer/Dockerfile)
* [requirements.txt](../.devcontainer/requirements.txt)

An image is built via the command

```shell
docker build . --file .devcontainer/Dockerfile -t text
```

On success, the output of

```shell
docker images
```

should include

<br>

| repository | tag    | image id | created  | size     |
|:-----------|:-------|:---------|:---------|:---------|
| text       | latest | $\ldots$ | $\ldots$ | $\ldots$ |


<br>

Subsequently, run a container, i.e., an instance, of the image `text` via:

<br>

```shell
docker run --rm --gpus all --shm-size=16gb -i -t 
  -p 127.0.0.1:6007:6007 -p 127.0.0.1:6006:6006 
    -p 172.17.0.2:8265:8265 -p 172.17.0.2:6379:6379 -w /app 
	    --mount type=bind,src="$(pwd)",target=/app 
	      -v ~/.aws:/root/.aws text
```

or

```shell
docker run --rm --gpus all --shm-size=16gb -i -t 
  -p 6007:6007 -p 6006:6006 -p 8265:8265 -p 6379:6379  
    -w /app --mount type=bind,src="$(pwd)",target=/app 
      -v ~/.aws:/root/.aws text
```

<br>

Herein, `-p 6007:6007` maps the host port `6007` to container port `6007`.  Note, the container's working environment, i.e., -w, must be inline with this project's top directory.  Additionally, visit the links below for more about the flags/options $\rightarrow$

* --rm: [automatically remove container](https://docs.docker.com/engine/reference/commandline/run/#:~:text=a%20container%20exits-,%2D%2Drm,-Automatically%20remove%20the)
* -i: [interact](https://docs.docker.com/engine/reference/commandline/run/#:~:text=and%20reaps%20processes-,%2D%2Dinteractive,-%2C%20%2Di)
* -t: [tag](https://docs.docker.com/get-started/02_our_app/#:~:text=Finally%2C%20the-,%2Dt,-flag%20tags%20your)
* -p: [publish the container's port/s to the host](https://docs.docker.com/engine/reference/commandline/run/#:~:text=%2D%2Dpublish%20%2C-,%2Dp,-Publish%20a%20container%E2%80%99s)
* --mount type=bind: [a bind mount](https://docs.docker.com/engine/storage/bind-mounts/#syntax)
* -v: [volume](https://docs.docker.com/engine/storage/volumes/)

<br>

The part `-v ~/.aws:/root/.aws` ascertains Amazon Web Services interactions via containers. Get the name of a running instance of ``points`` via:

```shell
docker ps --all
```

Never deploy a root container, study the production [Dockerfile](../Dockerfile); cf. remote [.devcontainer/Dockerfile](../.devcontainer/Dockerfile)

<br>

### Remote Development & Integrated Development Environments

An IDE (integrated development environment) is a helpful remote development tool.  The **IntelliJ
IDEA** set up involves connecting to a machine's Docker [daemon](https://www.jetbrains.com/help/idea/docker.html#connect_to_docker), the steps are

<br>

> * **Settings** $\rightarrow$ **Build, Execution, Deployment** $\rightarrow$ **Docker** $\rightarrow$ **WSL:** {select the linux operating system}
> * **View** $\rightarrow$ **Tool Window** $\rightarrow$ **Services** <br>Within the **Containers** section connect to the running instance of interest, or ascertain connection to the running instance of interest.

<br>

**Visual Studio Code** has its container attachment instructions; study [Attach Container](https://code.visualstudio.com/docs/devcontainers/attach-container).


<br>
<br>


## Code Analysis

The GitHub Actions script [main.yml](../.github/workflows/main.yml) conducts code analysis within a Cloud GitHub Workspace.  Depending on the script, code analysis may occur `on push` to any repository branch, or `on push` to a specific branch.  The sections herein outline remote code analysis.

### pylint

The directive

```shell
pylint --generate-rcfile > pylintrc
```

generates the dotfile `pylintrc` of the static code analyser [pylint](https://pylint.pycqa.org/en/latest/user_guide/checkers/features.html).  Analyse a directory via the command

```shell
python -m pylint --rcfile pylintrc {directory}
```

The `pylintrc` file of this template project has been **amended to adhere to team norms**, including

* Maximum number of characters on a single line.
  > max-line-length=127

* Maximum number of lines in a module.
  > max-module-lines=135


<br>

### pytest & pytest coverage

The directive patterns

```shell
python -m pytest tests/{directory.name}/...py
pytest --cov-report term-missing  --cov src/{directory.name}/...py tests/{directory.name}/...py
```

for test and test coverage, respectively.

<br>

### flake8

For code & complexity analysis.  A directive of the form

```bash
python -m flake8 --count --select=E9,F63,F7,F82 --show-source --statistics src/...
```

inspects issues in relation to logic (F7), syntax (Python E9, Flake F7), mathematical formulae symbols (F63), undefined variable names (F82).  Additionally

```shell
python -m flake8 --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics src/...
```

inspects complexity.


<br>
<br>

<br>
<br>

<br>
<br>

<br>
<br>
