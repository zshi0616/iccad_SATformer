# SATformer: Transformer-Based UNSAT Core Learning

Code repository for the paper:  
**SATformer: Transformer-Based UNSAT Core Learning**, ICCAD, under review.
Anonymous author(s)

## Abstract 
This paper introduces SATformer, a novel Transformer-based approach for the Boolean Satisfiability (SAT) problem. Rather than solving the problem directly, SATformer approaches the problem from the opposite direction by focusing on unsatisfiability. Specifically, it models clause interactions to identify any unsatisfiable sub-problems. Using a graph neural network, we convert clauses into clause embeddings and employ a hierarchical Transformer-based model to understand clause correlation. SATformer is trained through a multi-task learning approach, using the single-bit satisfiability result and the minimal unsatisfiable core (MUC) for UNSAT problems as clause supervision. 
As an end-to-end learning-based satisfiability classifier, the performance of SATformer surpasses that of NeuroSAT significantly. Furthermore, we integrate the clause predictions made by SATformer into modern heuristic-based SAT solvers and validate our approach with a logic equivalence checking task. Experimental results show that our SATformer can decrease the runtime of existing solvers by an average of 21.33%. 

## Installation
```sh
conda create -n satformer python=3.8.9
conda activate satformer
pip install -r requirements.txt
```

## Model Training 
Execute the bash script `./run/train.sh` to start training SATformer. You can change the configurations in this script. Please refer `./src/config.py` for the arguments
```sh
bash ./run/train.sh
```

## Model Testing 
Execute the bash script `./run/test.sh` to validate SATformer. 
```sh
bash ./run/test.sh
```