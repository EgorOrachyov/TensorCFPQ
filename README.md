# TensorCFPQ

Context free path querying on GPGPU powered by Nvidia CUDA.
Algorithm is based on tesor (Kronecker) product of recursive
automata for CFG and graph matrix.

Algo was originally developed by YaccConstructor [group](https://github.com/YaccConstructor).

## How to build and run

The following command line instructions allow you to clone this repo,
build test and run on test data.

```
$ git clone https://github.com/EgorOrachyov/TensorCFPQ.git
$ mkdir build
$ cd build
$ cmake ..
$ make 
$ ./tensor_cfpq
```

## Test data

All the original test data is hosted [here](https://github.com/JetBrains-Research/CFPQ_Data). 
Data for this repo test is stored in /data folder.

## Results

All the test were ran on the test machine with Intel(R) Core(TM) 4 Core i7-6700 CPU @ 3.40GHz with 32129 MiB RAM and Nvidia(R) GeForce(R) GTX 1070 TI with 8110 MiB VRAM.  

Recursive automata for CFG is stored in the automata.txt file in the same folder, as the 
test graphs group data. CFGs were converted manually. Original CFGs are in CNF or EBNF form. 

### Worst Case graphs

Test graph file | Time (in seconds) | S-symbol count
--------------- | ----------------- | --------------
worstcase_64.txt | 0.226068  | 1056
worstcase_128.txt | 1.997373  | 4160
worstcase_256.txt | 22.076006 | 16512

### RDF graphs

Test graph file | Time (in seconds) | S-symbol count
--------------- | ----------------- | --------------
atom-primitive.txt | 0.004649 | 15454
funding.txt | 0.068189 | 17634
pizza.txt | 0.042539 | 56195
biomedical-mesure-primitive.txt | 0.006330 | 15156
generations.txt | 0.001417 | 2164
core.txt | 0.756848 | 316
travel.txt | 0.002293 | 2499
univ-bench.txt | 0.001894 | 2540
wine.txt | 0.099101 | 66572

### Full Graph

Test graph file | Time (in seconds) | S-symbol count
--------------- | ----------------- | --------------
fullgraph_100.txt | 0.013094 | 10000
fullgraph_200.txt | 0.064668 | 40000
fullgraph_500.txt | 1.128082 | 250000
fullgraph_1000.txt | 15.376145 | 1000000

### Memory-Aliases

Test graph file | Time (in seconds) | M-symbol count | V-symbol count
--------------- | ----------------- | -------------- | --------------
bzip2.txt | 0.516798 | 315 | 2258
ls.txt | 11.451538 | 854 | 13051
pr.txt | 1.181471 | 385 | 3077
wc.txt | 0.032833 | 156 | 680

## Also

If you have any questions, feel free to contact me at egororachyov@gmail.com.


