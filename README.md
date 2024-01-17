# Tangle2Consensus
 
This is an IN DEVELOPMENT simulator made to simulate the Leaderless Nakamoto Consensus protocol outlined in: 

S. Müller, A. Penzkofer, N. Polyanskii, J. Theis, W. Sanders and H. Moog, "Tangle 2.0 Leaderless Nakamoto Consensus on the Heaviest DAG," in IEEE Access, vol. 10, pp. 105807-105842, 2022, doi: 10.1109/ACCESS.2022.3211422.

Current Classes:

Node Tangle:
Keeps track of all nodes capable of issuing to the Tangle and monitors and assigns witness weights to the nodes.
Future functionality - Add write and issue rate constraints to resist sybil attacks.

Node:
Capable of issuing blocks to the main block tangle. Keeps trackc of each node's signature and ID

Block Tangle: 
Monitors and records all blocks issued to the tangle. 

Block:
Contains a transaction, signature and references.
