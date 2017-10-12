# NRNNMF ------ Neighborhood regularized neural network matrix factorization

### 1. Introduction of file

|_ exam_code

​	|_ data

​		|_ adjacency_matrix.txt

​		|_ drug_similarity.txt

​		|_ gene_similarity.txt

​	|_nrnnmf

​		|_ __init\_\_.py

​		|_ model.py

​		|_ utils.py

​	|_ main.py

### 2. How to run

**i)** If you want to run the code with default arguments, you can type in the terminal as follows:

​	**python main.py**

Note: you must run this code in linux with Python2.x or python3.x and require several other Python packages, including Numpy(1.12.1), pandas(0.19.2), scikit-learn(0.18.1), Scipy(0.19.0)  and tensorflow(1.0.1). In addiction, it will cost you more than one hour.



**ii)** To get the results of different methods, please run main.py by setting suitable values for the following parameters:

--adjacency:			[str] the path and name of adjacency matrix name

--drug:				[str] the path and name of drug similarity file

--gene:				[str] the path and name of gene similarity file

--batch:				[int] the batch size for optimization using batch gradient descent(default = 10000)

--no-early:			[bool] indicate whether use early stopping(default = False)

--early-stop-max-iter:	[int] the maximum number of iterations for early stopping(default = 40)

--max-iters:			[int] the maximum number of iterations for optimization(default = 10000)



### 3. Reference

**i)** [https://github.com/stephenliu0423/PyDTI.git](https://github.com/stephenliu0423/PyDTI.git)

**ii)**https://github.com/jstol/neural-net-matrix-factorization