# 0: agent-time steps timing run
qsub sge.sh 

# 1: agent-time steps testing run
qsub sge.sh res_fast100_262144.sh


# 2: 75ms timing run
qsub sge.sh res_fast100_75ms.sh 1

# 3: 75ms testing run
qsub sge.sh res_fast100_75ms.sh 1024
