# Distributed-Tensorflow - Fashion-MNIST

## Instructions
1. In order to run this ditributed example, from 4 different terminals we execute the 4 commands for dispatching each of the tasks: 
 * For Parameter Server 0, ```python distributed_tensorflow_test.py --job_name='ps' --task_index=0```
 * For Worker 0          , ```python distributed_tensorflow_test.py --job_name='worker' --task_index=0```
 * For Worker 1          , ```python distributed_tensorflow_test.py --job_name='worker' --task_index=1```
 * For Worker 2          , ```python distributed_tensorflow_test.py --job_name='worker' --task_index=2```

2. Alternatively, the following will dispatch the four tasks automatically (depending on the system you are using, the output may all go to a single terminal or to 4 separate ones):
* For Parameter Server 0, ```subprocess.Popen('python distributed_tensorflow_test.py --job_name='ps' --task_index=0',shell= True)```
* For Worker 0          , ```subprocess.Popen('python distributed_tensorflow_test.py --job_name='worker' --task_index=0',shell= True)```
* For Worker 1          , ```subprocess.Popen('python distributed_tensorflow_test.py --job_name='worker' --task_index=1',shell= True)```
* For Worker 2          , ```subprocess.Popen('python distributed_tensorflow_test.py --job_name='worker' --task_index=2',shell= True)```

 > Remember to edit flag ps_hosts/worker_hosts inside the python sript before running th program  
 > Inside the script, flag num_workers/num_parameter_servers should be consistent with length of flag ps_hosts/worker_hosts

Training starts after 4 commands are executed. Fashion MNIST dataset is downloaded to `/data/fashion` directory and trained model is saved in  `/tmp/train_logs` directory. These directories can be changed with the `--data_dir` option and `--log_dir` option respectively.

### When trying on multiple machines

Execution method is not much different from that on a single machine, but there are the following changes.

* Change localhost to another Hostname or IP address.
* `--log_dir` must specify a shared directory accessible to all hosts making up the cluster.

## Dependency
 * [TensorFlow](https://www.tensorflow.org)
